"""
Utility functions to link Roboflow images to their corresponding JSON ground truth files.

Roboflow images have names like: IMG_20260110_111317_jpg.rf.aa5e3a0b94394a27cb167b3d45d1d4ad.jpg
Original images have names like: IMG_20260110_111317.jpg
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import ast


def scan_images_with_json(directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    Recursively scan a directory for images and link them to their JSON files.
    
    For each image (png/jpg), looks for:
    1. A JSON file with the same name (e.g., "image.jpg" -> "image.json")
    2. If not found, falls back to "generic.json" in the same folder
    
    Args:
        directory: Path to the directory to scan recursively
        
    Returns:
        Dictionary mapping image file path to:
        {
            "json_path": str,  # Path to the associated JSON file
            "json_data": Any   # Loaded JSON content
        }
        
    Example:
        >>> result = scan_images_with_json(Path("/path/to/images"))
        >>> result["/path/to/images/subfolder/photo.jpg"]
        {"json_path": "/path/to/images/subfolder/photo.json", "json_data": {...}}
    """
    result = {}
    directory = Path(directory)
    
    # Recursively find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for img_path in directory.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix not in image_extensions:
            continue
        
        # Look for a JSON with the same name
        json_same_name = img_path.with_suffix('.json')
        
        # Look for generic.json in the same folder
        generic_json = img_path.parent / "generic.json"
        
        json_path = None
        if json_same_name.exists():
            json_path = json_same_name
        elif generic_json.exists():
            json_path = generic_json
        
        if json_path is not None:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            result[str(img_path)] = {
                "json_path": str(json_path),
                "json_data": json_data
            }
        else:
            print(f"image {img_path} no json match found")
    
    return result


def parse_config_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a Python file containing 'code' and 'config' variables.
    
    Args:
        file_path: Path to the Python file containing code and config definitions
    
    Returns:
        Dictionary with 'code' and 'config' keys
        
    Example:
        >>> result = parse_config_file(Path("data.py"))
        >>> result['code']
        {'e': 'empty', 'br': 'brown', ...}
        >>> result['config']
        ['e', 'e', 'e', 'c', ...]
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Parse the file as Python code
    tree = ast.parse(content)
    
    result = {}
    
    # Extract variable assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if var_name in ['code', 'config']:
                        # Safely evaluate the value
                        result[var_name] = ast.literal_eval(node.value)
    
    return result


def extract_original_name(roboflow_filename: str) -> str:
    """
    Extract the original image name from a Roboflow filename.
    
    Args:
        roboflow_filename: Roboflow image name like "IMG_20260110_111317_jpg.rf.aa5e3a0b94394a27cb167b3d45d1d4ad.jpg"
    
    Returns:
        Original image name like "IMG_20260110_111317.jpg"
    
    Example:
        >>> extract_original_name("IMG_20260110_111317_jpg.rf.aa5e3a0b94394a27cb167b3d45d1d4ad.jpg")
        'IMG_20260110_111317.jpg'
    """
    # Split by ".rf." to get the part before the hash
    parts = roboflow_filename.split(".rf.")
    if len(parts) < 2:
        # Not a roboflow file, return as-is
        return roboflow_filename
    
    base_name = parts[0]
    
    # Replace the last "_jpg" or "_png" with the actual extension
    if base_name.endswith("_jpg"):
        return base_name[:-4] + ".jpg"
    elif base_name.endswith("_png"):
        return base_name[:-4] + ".png"
    
    return base_name


def scan_ground_truth_folder(ground_truth_dir: Path) -> Dict[str, Tuple[Path, List[str]]]:
    """
    Scan ground truth directory and map image names to their JSON files.
    
    Args:
        ground_truth_dir: Path to directory containing subfolders with images and JSON files
    
    Returns:
        Dictionary mapping image basename (without extension) to tuple of (json_path, [image_paths])
        
    Example:
        {
            "IMG_20260110_111317": (
                Path("/path/to/folder/data.json"),
                ["IMG_20260110_111317.jpg"]
            )
        }
    """
    image_to_json = {}
    
    # Iterate through all subdirectories
    for subfolder in ground_truth_dir.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Look for JSON files in this subfolder
        json_files = list(subfolder.glob("*.json"))
        
        if not json_files:
            # No JSON file in this folder, skip
            continue
        
        # Assume one JSON per folder (take the first if multiple)
        json_path = json_files[0]
        
        # List all images (jpg or png) in this folder
        image_files = []
        image_files.extend(subfolder.glob("*.jpg"))
        image_files.extend(subfolder.glob("*.JPG"))
        image_files.extend(subfolder.glob("*.png"))
        image_files.extend(subfolder.glob("*.PNG"))
        
        # Map each image to this JSON
        for img_path in image_files:
            # Use stem (filename without extension) as key
            img_basename = img_path.stem
            image_to_json[img_basename] = (json_path, [str(img_path)])
    
    return image_to_json


def link_roboflow_to_json(
    ground_truth_dir: Path,
    roboflow_dir: Path
) -> Dict[str, Dict]:
    """
    Link Roboflow images to their corresponding JSON files.
    
    Args:
        ground_truth_dir: Path to directory with subfolders containing images and JSON files
        roboflow_dir: Path to directory containing Roboflow images
    
    Returns:
        Dictionary mapping roboflow image path to metadata:
        {
            "roboflow_path": "/path/to/roboflow/IMG_xxx.rf.hash.jpg",
            "original_name": "IMG_xxx.jpg",
            "json_path": "/path/to/ground_truth/folder/data.json",
            "original_images": ["/path/to/original/IMG_xxx.jpg"]
        }
    """
    # First, scan ground truth directory
    image_to_json = scan_ground_truth_folder(ground_truth_dir)
    
    # Now link roboflow images
    results = {}
    
    for roboflow_img in roboflow_dir.iterdir():
        if not roboflow_img.is_file():
            continue
        
        if roboflow_img.suffix.lower() not in ['.jpg', '.png']:
            continue
        
        # Extract original name
        original_name = extract_original_name(roboflow_img.name)
        original_basename = Path(original_name).stem
        
        # Look up in ground truth mapping
        if original_basename in image_to_json:
            json_path, original_images = image_to_json[original_basename]
            
            results[str(roboflow_img)] = {
                "roboflow_path": str(roboflow_img),
                "original_name": original_name,
                "json_path": str(json_path),
                "original_images": original_images
            }
    
    return results


def print_mapping_summary(mapping: Dict[str, Dict]):
    """Print a summary of the roboflow to JSON mapping."""
    print(f"Total Roboflow images linked: {len(mapping)}")
    
    # Count unique JSON files
    unique_jsons = set(item["json_path"] for item in mapping.values())
    print(f"Unique JSON files referenced: {len(unique_jsons)}")
    
    # Show a few examples
    print("\nExample mappings:")
    for i, (roboflow_path, data) in enumerate(mapping.items()):
        if i >= 3:  # Show first 3
            break
        print(f"\n  Roboflow: {Path(roboflow_path).name}")
        print(f"  Original: {data['original_name']}")
        print(f"  JSON: {Path(data['json_path']).parent.name}/{Path(data['json_path']).name}")


if __name__ == "__main__":
    # Example usage
    ground_truth_dir = Path("/mnt/c/Users/nicol/Documents/star_genius_images/my_photos/")
    roboflow_dir = Path("/home/nicolas/code/star_genius_solver/data/annotatedv4/train/images/")
    
    mapping = link_roboflow_to_json(ground_truth_dir, roboflow_dir)
    print_mapping_summary(mapping)
    
    # Optionally save to JSON
    output_file = Path("/home/nicolas/code/star_genius_solver/data/roboflow_to_json_mapping.json")
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to: {output_file}")
