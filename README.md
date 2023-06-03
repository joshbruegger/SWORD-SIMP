# PunchDetect

PunchDetect is my Bachelor thesis project, which aims to use a sliding-window approach to detect punches in high-resolution images of late-medieval panel paintings. The project uses YOLO-NAS for object detection.

## Installation

Use the setup.sh file to install the required packages, and download the dataset.

```bash
chmod +x setup.sh
./setup.sh
```

## On the cluster

To run the code on the cluster, use the following command:

```bash
chmod +x cluster_job.sh
sbatch cluster_job.sh
```
