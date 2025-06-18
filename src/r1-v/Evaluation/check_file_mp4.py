import json
import os
import subprocess
from tqdm import tqdm

def is_strict_mp4(file_path):
    """
    Check the video file's format information using ffprobe.
    If the 'format_name' contains "mp4", then the file meets the strict mp4 encoding requirements;
    otherwise, return False along with ffprobe's output information.
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return False, result.stderr
    try:
        info = json.loads(result.stdout)
        format_name = info.get("format", {}).get("format_name", "")
        tokens = [token.strip() for token in format_name.split(',')]
        if "mp4" in tokens:
            return True, result.stdout
        else:
            return False, result.stdout
    except Exception as e:
        return False, str(e)

def convert_to_mp4(input_file, output_file):
    """
    Use ffmpeg to convert the video to MP4 encoding.
    The output is saved as a temporary file, and if the conversion is successful,
    the temporary file replaces the output_file.
    A scale filter is added to ensure the output resolution dimensions are even,
    preventing errors from libx264.
    """
    temp_file = output_file + ".temp.mp4"
    command = [
        "ffmpeg",
        "-y",                     # Overwrite output file if it exists
        "-i", input_file,         # Input file
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure width and height are even numbers
        "-c:v", "libx264",        # Use libx264 for video encoding
        "-c:a", "aac",            # Use AAC for audio encoding
        temp_file
    ]
    print(f"Converting: {input_file} -> {output_file}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Conversion failed: {input_file}\n{result.stderr}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    else:
        os.replace(temp_file, output_file)
        print(f"Conversion succeeded: {output_file}")
        return True

def find_alternative(file_path):
    """
    If the file specified by file_path does not exist, try to find a file with the same base name
    but with a different extension in the same directory.
    """
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists(dir_name):
        return None
    for candidate in os.listdir(dir_name):
        candidate_base, candidate_ext = os.path.splitext(candidate)
        if candidate_base == base_name and candidate_ext.lower() != ".mp4":
            candidate_full = os.path.join(dir_name, candidate)
            if os.path.isfile(candidate_full):
                return candidate_full
    return None

def process_videos_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    checked_paths = set()  # Record the file paths that have been checked
    for item in tqdm(data, desc="Processing videos", unit="item"):
        file_path = item.get("path", "").strip()
        # Skip if the file has already been checked
        if file_path in checked_paths:
            continue
        checked_paths.add(file_path)
        
        if os.path.exists(file_path):
            strict, info = is_strict_mp4(file_path)
            if not strict:
                print(f"\nVideo does not meet strict mp4 encoding requirements: {file_path}")
                print("ffprobe output:")
                print(info)
                # Convert the existing file to mp4 encoding (overwrite)
                convert_to_mp4(file_path, file_path)
        else:
            # Try to find an alternative file with the same base name but different extension
            alternative_file = find_alternative(file_path)
            if alternative_file:
                print(f"\nFound alternative: {alternative_file}")
                # Convert the alternative file to mp4 and save with the desired file_path
                convert_to_mp4(alternative_file, file_path)
            else:
                print(f"File does not exist and no alternative found: {file_path}")

if __name__ == "__main__":
    # Change this to the path of your JSON file
    process_videos_from_json("eval_mvbench.json")