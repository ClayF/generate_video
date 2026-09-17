#!/usr/bin/env python3
"""
Generate Video API client with base64 encoding
Client for generating videos from images using RunPod's generate_video endpoint
"""

import os
import requests
import json
import time
import base64
from typing import Optional, Dict, Any, List, Union
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateVideoClient:
    def __init__(
        self,
        runpod_endpoint_id: str,
        runpod_api_key: str
    ):
        """
        Initialize Generate Video client
        
        Args:
            runpod_endpoint_id: RunPod endpoint ID
            runpod_api_key: RunPod API key
        """
        self.runpod_endpoint_id = runpod_endpoint_id
        self.runpod_api_key = runpod_api_key
        self.runpod_api_endpoint = f"https://api.runpod.ai/v2/{runpod_endpoint_id}/run"
        self.status_url = f"https://api.runpod.ai/v2/{runpod_endpoint_id}/status"
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {runpod_api_key}',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"GenerateVideoClient initialized - Endpoint: {runpod_endpoint_id}")
    
    def encode_file_to_base64(self, file_path: str) -> Optional[str]:
        """
        Encode file to base64
        
        Args:
            file_path: File path to encode
        
        Returns:
            Base64 encoded string or None (on failure)
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')
            
            logger.info(f"✅ File base64 encoding completed: {file_path}")
            return base64_data
            
        except Exception as e:
            logger.error(f"❌ File base64 encoding failed: {e}")
            return None
    
    def submit_job(self, input_data: Dict[str, Any]) -> Optional[str]:
        """
        Submit job to RunPod
        
        Args:
            input_data: API input data
        
        Returns:
            Job ID or None (on failure)
        """
        payload = {"input": input_data}
        
        try:
            logger.info(f"Submitting job to RunPod: {self.runpod_api_endpoint}")
            logger.info(f"Input data: {json.dumps(input_data, indent=2, ensure_ascii=False)}")
            
            response = self.session.post(self.runpod_api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            job_id = response_data.get('id')
            
            if job_id:
                logger.info(f"✅ Job submission successful! Job ID: {job_id}")
                return job_id
            else:
                logger.error(f"❌ Failed to receive Job ID: {response_data}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Job submission failed: {e}")
            return None
    
    def wait_for_completion(self, job_id: str, check_interval: int = 10, max_wait_time: int = 1800) -> Dict[str, Any]:
        """
        Wait for job completion
        
        Args:
            job_id: Job ID
            check_interval: Status check interval (seconds)
            max_wait_time: Maximum wait time (seconds)
        
        Returns:
            Job result dictionary
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                logger.info(f"⏱️ Checking job status... (Job ID: {job_id})")
                
                response = self.session.get(f"{self.status_url}/{job_id}", timeout=30)
                response.raise_for_status()
                
                status_data = response.json()
                status = status_data.get('status')
                
                if status == 'COMPLETED':
                    logger.info("✅ Job completed!")
                    return {
                        'status': 'COMPLETED',
                        'output': status_data.get('output'),
                        'job_id': job_id
                    }
                elif status == 'FAILED':
                    logger.error("❌ Job failed.")
                    return {
                        'status': 'FAILED',
                        'error': status_data.get('error', 'Unknown error'),
                        'job_id': job_id
                    }
                elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                    logger.info(f"🏃 Job in progress... (Status: {status})")
                    time.sleep(check_interval)
                else:
                    logger.warning(f"❓ Unknown status: {status}")
                    return {
                        'status': 'UNKNOWN',
                        'data': status_data,
                        'job_id': job_id
                    }
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Status check error: {e}")
                time.sleep(check_interval)
        
        logger.error(f"❌ Job wait timeout ({max_wait_time} seconds)")
        return {
            'status': 'TIMEOUT',
            'job_id': job_id
        }
    
    def save_video_result(self, result: Dict[str, Any], output_path: str) -> bool:
        """
        Save video file from job result
        
        Args:
            result: Job result dictionary
            output_path: File path to save
        
        Returns:
            Save success status
        """
        try:
            if result.get('status') != 'COMPLETED':
                logger.error(f"Job not completed: {result.get('status')}")
                return False
            
            output = result.get('output', {})
            video_b64 = output.get('video')
            
            if not video_b64:
                logger.error("Video data not found")
                return False
            
            # Create directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Decode base64 and save video
            decoded_video = base64.b64decode(video_b64)
            
            with open(output_path, 'wb') as f:
                f.write(decoded_video)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Video saved successfully: {output_path} ({file_size / (1024*1024):.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Video save failed: {e}")
            return False
    
    @staticmethod
    def expansion_settings(
        language: str = "zh",
        model: Optional[str] = None,
        device: str = "GPU",
        mmproj: str = "(Auto-detect)",
        max_retries: int = 3,
        save_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Build the prompt_expansion block understood by handler.py.

        Args:
            language: 'zh' (recommended for Wan2.2), 'en' or 'auto'
            model: None = the worker's bundled local GGUF; a .gguf filename under
                   models/LLM; or a DashScope model name such as 'qwen3.7-plus'
                   (requires DASHSCOPE_API_KEY on the endpoint)
            device: 'GPU' or 'CPU' for local GGUF models
            mmproj: mmproj selection for local models ('(Auto-detect)' by default)
            max_retries: cloud API retry budget (1-10)
            save_tokens: compress the image before sending it to a cloud model
        """
        settings: Dict[str, Any] = {
            "language": language,
            "device": device,
            "mmproj": mmproj,
            "max_retries": max_retries,
            "save_tokens": save_tokens,
        }
        if model:
            settings["model"] = model
        return settings

    def expand_prompt(
        self,
        image_path: str,
        prompt: str,
        width: int = 480,
        height: int = 832,
        prompt_expansion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run only the prompt expansion on the worker and return the rewritten
        prompt (no video is generated). Handy for previewing/editing the prompt
        before committing GPU time to a full render.

        Returns the job result; on success result['output']['expanded_prompt']
        holds the text.
        """
        return self.create_video_from_image(
            image_path=image_path,
            prompt=prompt,
            width=width,
            height=height,
            prompt_expansion=prompt_expansion or self.expansion_settings(),
            expand_only=True,
        )

    def create_video_from_image(
        self,
        image_path: str,
        prompt: str = "running man, grab the gun",
        negative_prompt: Optional[str] = None,
        width: int = 480,
        height: int = 832,
        length: int = 81,
        steps: int = 10,
        seed: int = 42,
        cfg: float = 2.0,
        context_overlap: int = 48,
        lora_pairs: Optional[List[Dict[str, Any]]] = None,
        end_image_path: Optional[str] = None,
        prompt_expansion: Optional[Dict[str, Any]] = None,
        expand_only: bool = False
    ) -> Dict[str, Any]:
        """
        Generate video from image
        
        Args:
            image_path: Image file path
            prompt: Prompt text
            negative_prompt: Negative prompt to exclude unwanted elements
            width: Output width
            height: Output height
            length: Number of frames
            steps: Number of steps
            seed: Seed value
            cfg: CFG scale
            context_overlap: Context overlap
            lora_pairs: LoRA settings list (max 4)
            end_image_path: Optional end frame (switches to the start+end workflow)
            prompt_expansion: None to send the prompt as-is, or a dict such as
                {"language": "zh", "device": "GPU"} (see expansion_settings()) to
                have the worker rewrite it with the Wan Video Prompt Generator
            expand_only: Only run the prompt expansion and return the text
        
        Returns:
            Job result dictionary. With prompt expansion the output also carries
            'expanded_prompt' (the text actually sent to Wan2.2).
        """
        # Check file existence
        if not os.path.exists(image_path):
            return {"error": f"Image file does not exist: {image_path}"}
        
        # Encode image to base64
        image_base64 = self.encode_file_to_base64(image_path)
        if not image_base64:
            return {"error": "Image base64 encoding failed"}
        
        end_image_base64 = None
        if end_image_path:
            if not os.path.exists(end_image_path):
                return {"error": f"End image file does not exist: {end_image_path}"}
            end_image_base64 = self.encode_file_to_base64(end_image_path)
            if not end_image_base64:
                return {"error": "End image base64 encoding failed"}
        
        # Process LoRA settings
        if lora_pairs is None:
            lora_pairs = []
        
        # Support up to 4 LoRAs
        lora_count = min(len(lora_pairs), 4)
        if len(lora_pairs) > 4:
            logger.warning(f"LoRA count is {len(lora_pairs)}. Only up to 4 LoRAs are supported. Using first 4 only.")
            lora_pairs = lora_pairs[:4]
        
        # Configure API input data
        input_data = {
            "image_base64": image_base64,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "context_overlap": context_overlap,
            "lora_pairs": lora_pairs
        }
        
        # Add negative_prompt if provided
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt
        if end_image_base64:
            input_data["end_image_base64"] = end_image_base64
        if prompt_expansion is not None:
            input_data["prompt_expansion"] = prompt_expansion
        if expand_only:
            input_data["expand_only"] = True
        
        # Submit job and wait
        job_id = self.submit_job(input_data)
        if not job_id:
            return {"error": "Job submission failed"}
        
        result = self.wait_for_completion(job_id)
        return result
    
    def batch_process_images(
        self,
        image_folder_path: str,
        output_folder_path: str,
        valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        prompt: str = "running man, grab the gun",
        negative_prompt: Optional[str] = None,
        width: int = 480,
        height: int = 832,
        length: int = 81,
        steps: int = 10,
        seed: int = 42,
        cfg: float = 2.0,
        context_overlap: int = 48,
        lora_pairs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Batch process all image files in folder
        
        Args:
            image_folder_path: Folder path containing image files
            output_folder_path: Folder path to save results
            valid_extensions: Image file extensions to process
            prompt: Prompt text
            negative_prompt: Negative prompt to exclude unwanted elements
            width: Output width
            height: Output height
            length: Number of frames
            steps: Number of steps
            seed: Seed value
            cfg: CFG scale
            context_overlap: Context overlap
            lora_pairs: LoRA settings list
        
        Returns:
            Batch processing result dictionary
        """
        # Check path
        if not os.path.isdir(image_folder_path):
            return {"error": f"Image folder does not exist: {image_folder_path}"}
        
        # Create output folder
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Get image file list
        image_files = [
            f for f in os.listdir(image_folder_path)
            if f.lower().endswith(valid_extensions)
        ]
        
        if not image_files:
            return {"error": f"No image files to process: {image_folder_path}"}
        
        logger.info(f"Starting batch processing: {len(image_files)} files")
        
        results = {
            "total_files": len(image_files),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        # Process each image file
        for filename in image_files:
            logger.info(f"\n==================== Processing started: {filename} ====================")
            
            image_path = os.path.join(image_folder_path, filename)
            
            # Generate video
            result = self.create_video_from_image(
                image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                length=length,
                steps=steps,
                seed=seed,
                cfg=cfg,
                context_overlap=context_overlap,
                lora_pairs=lora_pairs
            )
            
            if result.get('status') == 'COMPLETED':
                # Save result file
                base_filename = os.path.splitext(filename)[0]
                output_filename = os.path.join(output_folder_path, f"result_{base_filename}.mp4")
                
                if self.save_video_result(result, output_filename):
                    logger.info(f"✅ [{filename}] Processing completed")
                    results["successful"] += 1
                    results["results"].append({
                        "filename": filename,
                        "status": "success",
                        "output_file": output_filename,
                        "job_id": result.get('job_id')
                    })
                else:
                    logger.error(f"[{filename}] Result save failed")
                    results["failed"] += 1
                    results["results"].append({
                        "filename": filename,
                        "status": "failed",
                        "error": "Result save failed",
                        "job_id": result.get('job_id')
                    })
            else:
                logger.error(f"[{filename}] Job failed: {result.get('error', 'Unknown error')}")
                results["failed"] += 1
                results["results"].append({
                    "filename": filename,
                    "status": "failed",
                    "error": result.get('error', 'Unknown error'),
                    "job_id": result.get('job_id')
                })
            
            logger.info(f"==================== Processing completed: {filename} ====================")
        
        logger.info(f"\n🎉 Batch processing completed: {results['successful']}/{results['total_files']} successful")
        return results


def main():
    """Usage example"""
    
    # Configuration (change to actual values)
    ENDPOINT_ID = "your-endpoint-id"
    RUNPOD_API_KEY = "your-runpod-api-key"
    
    # Initialize client
    client = GenerateVideoClient(
        runpod_endpoint_id=ENDPOINT_ID,
        runpod_api_key=RUNPOD_API_KEY
    )
    
    print("=== Generate Video Client Usage Example ===\n")
    
    # Example 1: Single image processing
    print("1. Single image processing")
    result1 = client.create_video_from_image(
        image_path="./example_image.png",
        prompt="running man, grab the gun",
        negative_prompt="blurry, low quality, distorted",
        width=480,
        height=832,
        length=81,
        steps=10,
        seed=42,
        cfg=2.0
    )
    
    if result1.get('status') == 'COMPLETED':
        client.save_video_result(result1, "./output_single.mp4")
    else:
        print(f"Error: {result1.get('error')}")
    
    print("\n" + "-"*50 + "\n")
    
    # Example 1b: Prompt expansion (Wan Video Prompt Generator runs on the worker)
    print("1b. Prompt expansion")
    preview = client.expand_prompt(
        image_path="./example_image.png",
        prompt="running man, grab the gun",
        prompt_expansion=client.expansion_settings(language="zh"),
    )
    if preview.get('status') == 'COMPLETED':
        print("Expanded prompt:", preview['output']['expanded_prompt'])
    result1b = client.create_video_from_image(
        image_path="./example_image.png",
        prompt="running man, grab the gun",
        width=480,
        height=832,
        length=81,
        seed=42,
        cfg=2.0,
        prompt_expansion=client.expansion_settings(language="zh", device="GPU"),
    )
    if result1b.get('status') == 'COMPLETED':
        print("Prompt used for the video:", result1b['output'].get('expanded_prompt'))
        client.save_video_result(result1b, "./output_expanded.mp4")
    else:
        print(f"Error: {result1b.get('error')}")
    
    print("\n" + "-"*50 + "\n")
    
    # Example 2: Processing with LoRA
    print("2. Processing with LoRA")
    lora_pairs = [
        {
            "high": "your_high_lora.safetensors",
            "low": "your_low_lora.safetensors",
            "high_weight": 1.0,
            "low_weight": 1.0
        }
    ]
    
    result2 = client.create_video_from_image(
        image_path="./example_image.png",
        prompt="running man, grab the gun",
        negative_prompt="blurry, low quality, distorted",
        width=480,
        height=832,
        length=81,
        steps=10,
        seed=42,
        cfg=2.0,
        lora_pairs=lora_pairs
    )
    
    if result2.get('status') == 'COMPLETED':
        client.save_video_result(result2, "./output_lora.mp4")
    else:
        print(f"Error: {result2.get('error')}")
    
    print("\n" + "-"*50 + "\n")
    
    # Example 3: Batch processing (uncomment to use)
    # print("3. Batch processing")
    # batch_result = client.batch_process_images(
    #     image_folder_path="./input_images",
    #     output_folder_path="./output_videos",
    #     prompt="running man, grab the gun",
    #     width=480,
    #     height=832,
    #     length=81,
    #     steps=10,
    #     seed=42,
    #     cfg=2.0
    # )
    
    # print(f"Batch processing result: {batch_result}")
    
    print("\n=== All examples completed ===")


if __name__ == "__main__":
    main()
