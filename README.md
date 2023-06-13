# PunchDetect

PunchDetect is my Bachelor thesis project, which aims to use a sliding-window approach to detect punches in high-resolution images of late-medieval panel paintings. The project uses YOLO-NAS for object detection.

## Installation

Use the setup.sh file to install the required packages, and download the dataset.

1. Clone the repository
```bash
git clone https://github.com/joshbruegger/SWORD-SIMP
```
2. Give the setup.sh file execution rights
```bash
chmod +x setup.sh
```
3. Run the setup.sh file

```bash 
sbatch setup.sh
```

The setup script has the following usage and options:
```bash
setup_dataset.sh [-d] [-b] [-c] [-e] [-n <number>]
```
- d: force download of dataset
- b: force generation of bboxes
- c: force generation of crops
- e: force generation of environment
- n <number>: number of crops to generate (default = 10)

combination of flags is possible (e.g. -bc), except for -n.
