# Prediction interface for Cog ⚙️
# https://cog.run/python

import glob
import os
import random
import subprocess
import sys
import tempfile
from collections import OrderedDict

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from inference import (
    DeepSpeech,
    LipSick,
    LipSickInferenceOptions,
    compute_crop_radius,
    convert_audio_to_wav,
    extract_frames_from_video,
    get_versioned_filename,
    load_landmark_dlib,
    parse_reference_indices,
)
from utils.blend import main as blend_video


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Using device: {device}")

        opt = LipSickInferenceOptions().parse_args()

        print("Loading DeepSpeech model...")
        DSModel = DeepSpeech(opt.deepspeech_model_path)
        self.DSModel = DSModel
        print("Done!")

        print("Loading LipSick model...")
        model = LipSick(opt.source_channel, opt.ref_channel, opt.audio_channel).to(
            device
        )
        if not os.path.exists(opt.pretrained_lipsick_path):
            raise Exception(
                f"Wrong path of pretrained model weight: {opt.pretrained_lipsick_path}"
            )
        state_dict = torch.load(
            opt.pretrained_lipsick_path, map_location=torch.device("cpu")
        )["state_dict"]["net_g"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        self.model = model
        print("Done!")

    def predict(
        self,
        source_video: Path = Input(description="source video"),
        driving_audio: Path = Input(description="driving audio"),
        mouth_region_size: int = Input(
            description="Size of the mouth region.", default=256
        ),
        custom_reference_frames: str = Input(
            description="Comma-separated list of custom reference frame indices.",
            default="",
        ),
        custom_crop_radius: int = Input(
            description="Custom crop radius for all frames.", default=None
        ),
        auto_mask: bool = Input(
            description="Generate a same-length video for auto masking.", default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        print(f"source_video            : {source_video}")
        print(f"driving_audio           : {driving_audio}")
        print(f"mouth_region_size       : {mouth_region_size}")
        print(f"custom_reference_frames : {custom_reference_frames}")
        print(f"custom_crop_radius      : {custom_crop_radius}")
        print(f"auto_mask               : {auto_mask}")

        source_video = str(source_video)
        driving_audio = str(driving_audio)

        job_dir = tempfile.mkdtemp(prefix="LipSick-")

        if not source_video:
            raise Exception("Source video not provided!")
        if not driving_audio:
            raise Exception("Driving audio not provided!")

        driving_audio = convert_audio_to_wav(driving_audio)

        print("Extracting frames from video")
        video_frame_dir = os.path.join(job_dir, "frames")
        if not os.path.exists(video_frame_dir):
            os.mkdir(video_frame_dir)
        video_size = extract_frames_from_video(source_video, video_frame_dir)

        ds_feature = self.DSModel.compute_audio_feature(driving_audio)
        res_frame_length = ds_feature.shape[0]
        ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode="edge")

        print("Tracking face")
        video_frame_path_list = glob.glob(os.path.join(video_frame_dir, "*.jpg"))
        video_frame_path_list.sort()
        video_landmark_data = np.array(
            [load_landmark_dlib(frame) for frame in video_frame_path_list]
        )

        print("Aligning frames with driving audio")
        video_frame_path_list_cycle = (
            video_frame_path_list + video_frame_path_list[::-1]
        )
        video_landmark_data_cycle = np.concatenate(
            [video_landmark_data, np.flip(video_landmark_data, 0)], 0
        )
        video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
        if video_frame_path_list_cycle_length >= res_frame_length:
            res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
            res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
        else:
            divisor = res_frame_length // video_frame_path_list_cycle_length
            remainder = res_frame_length % video_frame_path_list_cycle_length
            res_video_frame_path_list = (
                video_frame_path_list_cycle * divisor
                + video_frame_path_list_cycle[:remainder]
            )
            res_video_landmark_data = np.concatenate(
                [video_landmark_data_cycle] * divisor
                + [video_landmark_data_cycle[:remainder, :, :]],
                0,
            )
        res_video_frame_path_list_pad = (
            [video_frame_path_list_cycle[0]] * 2
            + res_video_frame_path_list
            + [video_frame_path_list_cycle[-1]] * 2
        )
        res_video_landmark_data_pad = np.pad(
            res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode="edge"
        )
        assert (
            ds_feature_padding.shape[0]
            == len(res_video_frame_path_list_pad)
            == res_video_landmark_data_pad.shape[0]
        )
        pad_length = ds_feature_padding.shape[0]

        ref_img_list = []
        resize_w = int(mouth_region_size + mouth_region_size // 4)
        resize_h = int((mouth_region_size // 2) * 3 + mouth_region_size // 8)
        if custom_reference_frames:
            ref_index_list = parse_reference_indices(custom_reference_frames)
            print(f"Selecting reference images based on input: {ref_index_list}")
        else:
            ref_index_list = random.sample(
                range(5, len(res_video_frame_path_list_pad) - 2), 5
            )
            print(f"Selecting reference images randomly: {ref_index_list}")

        # print("If each value has +5 added do not be alarmed it will -5 later")
        for ref_index in ref_index_list:
            if custom_crop_radius and custom_crop_radius > 0:
                crop_radius, crop_flag = custom_crop_radius, True
            else:
                crop_flag, crop_radius = compute_crop_radius(
                    video_size,
                    res_video_landmark_data_pad[ref_index - 5 : ref_index, :, :],
                )

            crop_radius_1_4 = crop_radius // 4
            ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[
                :, :, ::-1
            ]
            ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
            ref_img_crop = ref_img[
                ref_landmark[29, 1]
                - crop_radius : ref_landmark[29, 1]
                + crop_radius * 2
                + crop_radius // 4,
                ref_landmark[33, 0]
                - crop_radius
                - crop_radius // 4 : ref_landmark[33, 0]
                + crop_radius
                + crop_radius // 4,
            ]
            ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h))
            ref_img_crop = ref_img_crop / 255.0
            ref_img_list.append(ref_img_crop)
        ref_video_frame = np.concatenate(ref_img_list, axis=2)
        ref_img_tensor = (
            torch.from_numpy(ref_video_frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        res_video_name = "facial_dubbing.mp4"
        res_video_path = os.path.join(job_dir, res_video_name)
        res_video_path = get_versioned_filename(
            res_video_path
        )  # Ensure unique filename
        videowriter = cv2.VideoWriter(
            res_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, video_size
        )

        if auto_mask:
            samelength_video_name = "samelength.mp4"
            samelength_video_path = os.path.join(job_dir, samelength_video_name)
            samelength_video_path = get_versioned_filename(
                samelength_video_path
            )  # Ensure unique filename
            videowriter_samelength = cv2.VideoWriter(
                samelength_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, video_size
            )

        res_face_name = "facial_dubbing_face.mp4"
        res_face_path = os.path.join(job_dir, res_face_name)
        res_face_path = get_versioned_filename(res_face_path)  # Ensure unique filename
        videowriter_face = cv2.VideoWriter(
            res_face_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (resize_w, resize_h)
        )

        for clip_end_index in range(5, pad_length, 1):
            sys.stdout.write(
                f"\rSynthesizing {clip_end_index - 5}/{pad_length - 5} frame"
            )
            sys.stdout.flush()  # Make sure to flush the output buffer
            if not crop_flag:
                crop_radius = compute_crop_radius(
                    video_size,
                    res_video_landmark_data_pad[
                        clip_end_index - 5 : clip_end_index, :, :
                    ],
                    random_scale=1.10,
                )

            crop_radius_1_4 = crop_radius // 4
            frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[
                :, :, ::-1
            ]
            frame_data_samelength = frame_data.copy()
            if auto_mask:
                videowriter_samelength.write(frame_data_samelength[:, :, ::-1])
            frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
            crop_frame_data = frame_data[
                frame_landmark[29, 1]
                - crop_radius : frame_landmark[29, 1]
                + crop_radius * 2
                + crop_radius_1_4,
                frame_landmark[33, 0]
                - crop_radius
                - crop_radius_1_4 : frame_landmark[33, 0]
                + crop_radius
                + crop_radius_1_4,
            ]
            crop_frame_h, crop_frame_w = (
                crop_frame_data.shape[0],
                crop_frame_data.shape[1],
            )
            crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h)) / 255.0
            crop_frame_data[
                mouth_region_size // 2 : mouth_region_size // 2 + mouth_region_size,
                mouth_region_size // 8 : mouth_region_size // 8 + mouth_region_size,
                :,
            ] = 0

            crop_frame_tensor = (
                torch.from_numpy(crop_frame_data)
                .float()
                .to(self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            deepspeech_tensor = (
                torch.from_numpy(
                    ds_feature_padding[clip_end_index - 5 : clip_end_index, :]
                )
                .permute(1, 0)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                pre_frame = self.model(
                    crop_frame_tensor, ref_img_tensor, deepspeech_tensor
                )
                pre_frame = (
                    pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
                )
            videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
            pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))
            frame_data[
                frame_landmark[29, 1]
                - crop_radius : frame_landmark[29, 1]
                + crop_radius * 2,
                frame_landmark[33, 0]
                - crop_radius
                - crop_radius_1_4 : frame_landmark[33, 0]
                + crop_radius
                + crop_radius_1_4,
                :,
            ] = pre_frame_resize[: crop_radius * 3, :, :]
            videowriter.write(frame_data[:, :, ::-1])
        videowriter.release()
        if auto_mask:
            videowriter_samelength.release()
        videowriter_face.release()

        if auto_mask:
            video_add_audio_path = os.path.join(job_dir, "pre_blend.mp4")
        else:
            video_add_audio_path = os.path.join(
                job_dir,
                "LIPSICK.mp4",
            )

        cmd = f'ffmpeg -r 25 -i "{res_video_path}" -i "{driving_audio}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{video_add_audio_path}"'
        subprocess.call(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )  # Suppress FFmpeg logs
        # os.remove(res_video_path)  # Clean up intermediate files
        # os.remove(res_face_path)  # Clean up intermediate files

        if auto_mask:
            print("Auto Mask stage")
            # samelength_video_path = os.path.join(opt.res_video_dir, "samelength.mp4")
            pre_blend_video_path = video_add_audio_path

            res_file = blend_video(samelength_video_path, pre_blend_video_path)
        else:
            res_file = video_add_audio_path

        return Path(res_file)
