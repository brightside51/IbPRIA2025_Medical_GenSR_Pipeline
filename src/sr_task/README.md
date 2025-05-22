# Running the Deep Learning Super-Resolution Model

To run the developed deep learning super-resolution model, use the `model_generate_txt.py` script with the following arguments:

<path_before_relative> <input_file> <model_path> <output_directory> [scale]


## Arguments:

1. **path_before_relative**: The base path of the paths found in the `.txt` file.
2. **input_file**: The `.txt` file containing the paths to the input images.
3. **model_path**: The path to the model generator file (`.pth`).
4. **output_directory**: The path where the upscaled images will be saved.
5. **scale** (optional): The scaling factor. If not provided, the standard value is 4.

### Example Usage:

```bash
python model_generate_txt.py /base/path input_paths.txt model_generator.pth output_directory 4

