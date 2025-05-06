#!/usr/bin/env python3

import argparse
import os
import shutil
import sys 
import time
from pathlib import Path
from typing import Union 

import cv2
import numpy as np
# from PIL import Image # PIL is not strictly needed for this script's core logic now

# --- Python Path Modification for local mlx_clip_lib ---
script_dir = os.path.dirname(os.path.abspath(__file__))
mlx_lib_path = os.path.join(script_dir, 'mlx_clip_lib')

if os.path.isdir(mlx_lib_path):
    if os.path.isdir(os.path.join(mlx_lib_path, 'mlx_clip')):
        sys.path.insert(0, mlx_lib_path)
        print(f"Added {mlx_lib_path} to sys.path for local mlx_clip library.")
    else:
        print(f"Warning: Found '{mlx_lib_path}', but the inner 'mlx_clip' package directory is missing. "
              "Imports might fail.")
else:
    print(f"Warning: Local 'mlx_clip_lib' directory not found at {mlx_lib_path}. "
          "Assuming mlx_clip is installed globally or in PYTHONPATH.")
# --- End Python Path Modification ---

try:
    from mlx_clip import mlx_clip
    from mlx_clip import convert
    import mlx.core as mx
except ImportError as e:
    print(f"Initial ImportError: {e}")
    print("\nError: mlx_clip library components not found or import failed.")
    print("This script expects the 'mlx_clip' package (from harperreed/mlx_clip) to be accessible.")
    print("If you cloned it into 'mlx_clip_lib' next to this script, ensure that directory contains "
          "the 'mlx_clip' package folder (with __init__.py, mlx_clip.py, etc.).")
    print(f"Current sys.path relevant entries:\n{sys.path[:3]}") 
    print("Alternatively, install mlx_clip into your Python environment (e.g., `pip install .` "
          "from within the cloned harperreed/mlx_clip directory).")
    exit(1)

DEFAULT_MODEL_HF_REPO = "openai/clip-vit-base-patch32"
DEFAULT_MLX_MODEL_DIR = "mlx_models"

def ensure_model_is_available(hf_repo_id: str, mlx_model_base_path: str) -> str:
    model_name = hf_repo_id.split("/")[-1]
    mlx_model_path = Path(mlx_model_base_path) / model_name
    
    if not mlx_model_path.exists() or not any(mlx_model_path.iterdir()):
        print(f"MLX model not found at {mlx_model_path}. Converting from Hugging Face repo {hf_repo_id}...")
        mlx_model_path.mkdir(parents=True, exist_ok=True)
        try:
            convert.convert_weights(hf_repo_id, str(mlx_model_path))
            print(f"Model converted and saved to {mlx_model_path}")
        except Exception as e:
            print(f"Error converting model: {e}")
            print("Please ensure you have an internet connection and necessary permissions.")
            if mlx_model_path.exists():
                shutil.rmtree(mlx_model_path) # Clean up partially created model dir
            raise
    else:
        print(f"Using existing MLX model from {mlx_model_path}")
    return str(mlx_model_path)

def extract_frames(video_path: str, output_folder: str, interval_sec: float = 1.0, target_height: Union[int, None] = None) -> list:
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    # output_folder (temp_frames) is now cleared *before* calling this in main
    # So, we just ensure it exists.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Video FPS is 0. Cannot process.")
        cap.release()
        return []
        
    frames_to_skip = max(1, int(round(fps * interval_sec)))
    print(f"Video FPS: {fps:.2f}, extracting one frame every {frames_to_skip} video frames (approx. every {interval_sec:.2f}s).")
    if target_height:
        print(f"Extracted frames will be resized to a target height of {target_height}px (aspect ratio preserved).")


    saved_frames_info = []
    frame_count = 0 
    saved_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret or frame is None:
            # print(f"Warning: Could not read frame at index {frame_count} (total: {total_frames}).")
            break 

        frame_to_save = frame
        if target_height is not None:
            original_h, original_w = frame.shape[:2]
            if original_h > target_height: # Only downscale
                scale_ratio = float(target_height) / original_h
                new_w = int(round(original_w * scale_ratio))
                if new_w > 0: # Ensure new width is positive
                    resized_frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
                    frame_to_save = resized_frame
                # else: frame has extremely narrow aspect ratio or target_height is tiny. Keep original.
        
        timestamp_sec = frame_count / fps
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}_{timestamp_sec:.2f}s.png")
        cv2.imwrite(frame_filename, frame_to_save)
        saved_frames_info.append({"timestamp": timestamp_sec, "path": frame_filename})
        saved_count += 1
        
        frame_count += frames_to_skip

    cap.release()
    print(f"Extracted {saved_count} frames into '{output_folder}'.")
    return saved_frames_info

def get_image_embedding(clip_model_instance, image_path: str) -> mx.array:
    raw_output = clip_model_instance.image_encoder(image_path)
    if isinstance(raw_output, list):
        if all(isinstance(item, (int, float)) for item in raw_output):
            return mx.array(raw_output)[None, :] 
        else:
            raise TypeError(f"image_encoder for {Path(image_path).name} returned unexpected list format.")
    elif isinstance(raw_output, mx.array):
        if raw_output.ndim == 1: 
            return raw_output[None, :] 
        elif raw_output.ndim == 2 and raw_output.shape[0] == 1: 
            return raw_output
        else:
            raise ValueError(f"image_encoder for {Path(image_path).name} returned mx.array with unexpected shape: {raw_output.shape}")
    raise TypeError(f"image_encoder for {Path(image_path).name} returned unexpected type: {type(raw_output)}")

def get_text_embedding(clip_model_instance, text: str) -> Union[mx.array, list]:
    return clip_model_instance.text_encoder(text)

def find_nearest_frame(text_embedding: mx.array, image_embeddings_info: list) -> dict:
    if not image_embeddings_info:
        return None

    current_text_embedding = text_embedding
    if current_text_embedding.ndim == 1: 
        current_text_embedding = current_text_embedding[None, :] 
    elif not (current_text_embedding.ndim == 2 and current_text_embedding.shape[0] == 1):
        raise ValueError(
            f"Text embedding has unexpected shape: {current_text_embedding.shape}. Expected (D,) or (1,D)."
        )
    
    text_norm_val = mx.linalg.norm(current_text_embedding, axis=1, keepdims=True) 
    text_embedding_normalized = current_text_embedding / text_norm_val 
    
    try:
        raw_image_embeddings_stack = mx.stack([info["embedding"] for info in image_embeddings_info])
    except Exception as e:
        print("Error during mx.stack of image embeddings. Individual embedding shapes:")
        for i, info in enumerate(image_embeddings_info[:5]): 
            print(f"  Frame {i} path: {info['path']}, embedding type: {type(info['embedding'])}, shape: {info['embedding'].shape if hasattr(info['embedding'], 'shape') else 'N/A'}")
        raise e

    if not (raw_image_embeddings_stack.ndim == 3 and raw_image_embeddings_stack.shape[1] == 1):
        if raw_image_embeddings_stack.ndim == 2 and raw_image_embeddings_stack.shape[1] == current_text_embedding.shape[1]: 
            all_image_embeddings_mx = raw_image_embeddings_stack 
        else:
            raise ValueError(
                f"Stacked image embeddings have unexpected shape: {raw_image_embeddings_stack.shape}. Expected (N,1,D) or (N,D with D={current_text_embedding.shape[1]})."
            )
    else: 
        all_image_embeddings_mx = raw_image_embeddings_stack.squeeze(axis=1) 
    
    img_norms_val = mx.linalg.norm(all_image_embeddings_mx, axis=1, keepdims=True) 
    all_image_embeddings_norm = all_image_embeddings_mx / img_norms_val 
    
    similarities = mx.matmul(all_image_embeddings_norm, text_embedding_normalized.T) 
    similarities = similarities.squeeze() 
    
    mx.eval(similarities)

    best_idx = mx.argmax(similarities).item()
    
    best_match_info = image_embeddings_info[best_idx].copy()
    best_match_info["similarity"] = similarities[best_idx].item()
    
    return best_match_info

def main():
    parser = argparse.ArgumentParser(description="Find the nearest video frame to a text query using CLIP.")
    parser.add_argument("video_file", type=str, help="Path to the video file (e.g., eightandahalf.mp4)")
    parser.add_argument("text_query", type=str, help="Text query to search for in the video frames.")
    parser.add_argument("--model_repo", type=str, default=DEFAULT_MODEL_HF_REPO, 
                        help=f"Hugging Face model repo ID (default: {DEFAULT_MODEL_HF_REPO}). ")
    parser.add_argument("--mlx_model_dir", type=str, default=DEFAULT_MLX_MODEL_DIR, 
                        help=f"Directory to store/load MLX converted models (default: ./{DEFAULT_MLX_MODEL_DIR})")
    parser.add_argument("--frame_interval", type=float, default=1.0, help="Interval in seconds to extract frames (default: 1.0 second).")
    parser.add_argument("--target_frame_height", type=int, default=None,
                        help="Target height to resize extracted frames (e.g., 480). Preserves aspect ratio. Original size if not set. Only downscaling is performed.")
    parser.add_argument("--temp_frame_dir", type=str, default="temp_frames", help="Directory to store temporary frames from the current run.")
    parser.add_argument("--delete_frames_after", action="store_true", 
                        help="If set, delete the temporary frames directory after execution. Default is to keep them.")
    
    args = parser.parse_args()

    clip_model_instance = None
    try:
        mlx_converted_model_path = ensure_model_is_available(args.model_repo, args.mlx_model_dir)
        print(f"Loading CLIP model using path: {mlx_converted_model_path}...")
        clip_model_instance = mlx_clip(mlx_converted_model_path)
    except Exception as e:
        print(f"Failed to load or convert CLIP model: {e}")
        # No frame directory cleanup needed here as it hasn't been created for this run yet
        return

    # --- Frame Extraction Setup ---
    # ALWAYS clear temp_frames at the start of a run to ensure it only contains output from this run.
    if os.path.exists(args.temp_frame_dir):
        print(f"Clearing existing temporary frame directory '{args.temp_frame_dir}' for a fresh run.")
        shutil.rmtree(args.temp_frame_dir)
    # Path(args.temp_frame_dir).mkdir(parents=True, exist_ok=True) # extract_frames will create it

    print(f"\nExtracting new frames into '{args.temp_frame_dir}'...")
    extracted_frames_info = extract_frames(args.video_file, args.temp_frame_dir, args.frame_interval, args.target_frame_height)
    
    if not extracted_frames_info:
        print("No frames were extracted. Exiting.")
        # The temp_frame_dir might be empty or partially filled.
        # Final cleanup logic below will handle it based on --delete_frames_after.
        if args.delete_frames_after:
            if os.path.exists(args.temp_frame_dir):
                try:
                    shutil.rmtree(args.temp_frame_dir)
                    print(f"Cleaned up {args.temp_frame_dir} due to no frames extracted and --delete_frames_after.")
                except OSError as e_clean:
                    print(f"Error cleaning up {args.temp_frame_dir}: {e_clean}")
        else:
            print(f"Keeping {args.temp_frame_dir} (may be empty/partial) as --delete_frames_after is not set.")
        return

    print(f"\nEmbedding text query: '{args.text_query}'...")
    start_time = time.time()
    raw_text_embedding_output = get_text_embedding(clip_model_instance, args.text_query)
    
    text_embedding_final = None
    if isinstance(raw_text_embedding_output, list):
        if all(isinstance(item, (int, float)) for item in raw_text_embedding_output):
            # print(f"Note: text_encoder returned a Python list of {len(raw_text_embedding_output)} numbers. Converting to mx.array.")
            text_embedding_final = mx.array(raw_text_embedding_output) 
        elif len(raw_text_embedding_output) == 1 and isinstance(raw_text_embedding_output[0], mx.array):
            # print("Note: text_encoder returned a list containing a single mx.array element.")
            text_embedding_final = raw_text_embedding_output[0]
        else:
            list_content_summary = "empty"
            if raw_text_embedding_output: list_content_summary = f"first element type: {type(raw_text_embedding_output[0])}"
            raise TypeError(
                f"text_encoder returned an unexpected list format: <class 'list'> with length {len(raw_text_embedding_output)} "
                f"and content ({list_content_summary}). Expected list of numbers or list of one mx.array."
            )
    elif isinstance(raw_text_embedding_output, mx.array):
        text_embedding_final = raw_text_embedding_output
    else:
        raise TypeError(f"text_encoder returned an unexpected type: {type(raw_text_embedding_output)}")
    
    if not isinstance(text_embedding_final, mx.array): 
        raise TypeError(f"text_embedding_final is not an mx.array after processing. Type: {type(text_embedding_final)}")

    mx.eval(text_embedding_final) 
    print(f"Text embedding (shape: {text_embedding_final.shape}) generated in {time.time() - start_time:.2f}s.")

    print(f"\nEmbedding {len(extracted_frames_info)} image frames...")
    image_embeddings_with_info = []
    start_time = time.time()
    for i, frame_info in enumerate(extracted_frames_info):
        try:
            embedding = get_image_embedding(clip_model_instance, frame_info["path"])
            mx.eval(embedding) 
            image_embeddings_with_info.append({
                "timestamp": frame_info["timestamp"],
                "path": frame_info["path"],
                "embedding": embedding 
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(extracted_frames_info):
                print(f"  Embedded frame {i+1}/{len(extracted_frames_info)}")
        except Exception as e:
            print(f"Error embedding frame {frame_info['path']}: {e}")
    
    print(f"All image embeddings generated in {time.time() - start_time:.2f}s.")

    if not image_embeddings_with_info:
        print("No image embeddings could be generated. Exiting.")
        # Final cleanup logic below will handle it based on --delete_frames_after.
        if args.delete_frames_after:
            if os.path.exists(args.temp_frame_dir):
                try:
                    shutil.rmtree(args.temp_frame_dir)
                    print(f"Cleaned up {args.temp_frame_dir} due to no embeddings and --delete_frames_after.")
                except OSError as e_clean:
                    print(f"Error cleaning up {args.temp_frame_dir}: {e_clean}")
        else:
            print(f"Keeping {args.temp_frame_dir} (contains frames but no embeddings processed for them) as --delete_frames_after is not set.")
        return

    print("\nFinding the nearest frame to the text query...")
    start_time = time.time()
    best_match = find_nearest_frame(text_embedding_final, image_embeddings_with_info)
    print(f"Search completed in {time.time() - start_time:.2f}s.")

    if best_match:
        print("\n--- Best Match Found ---")
        print(f"Text Query: '{args.text_query}'")
        print(f"Best matching frame timestamp: {best_match['timestamp']:.2f} seconds")
        print(f"Frame path: {best_match['path']}")
        print(f"Similarity score: {best_match['similarity']:.4f}")
    else:
        print("Could not find a best match.")

    # --- Final Cleanup Logic ---
    if args.delete_frames_after: # Check the flag
        print(f"\nCleaning up temporary frame directory: {args.temp_frame_dir} as requested by --delete_frames_after.")
        try:
            if os.path.exists(args.temp_frame_dir):
                shutil.rmtree(args.temp_frame_dir)
                print("Temporary files deleted.")
            else: # Should not happen if frames were extracted
                print(f"Temporary frame directory {args.temp_frame_dir} not found for cleanup.")
        except OSError as e:
            print(f"Error deleting temporary directory {args.temp_frame_dir}: {e}")
    else:
        print(f"\nKeeping temporary frame directory: {args.temp_frame_dir} (contains frames from this run).")

if __name__ == "__main__":
    main()