English | [中文简体](README.zh-CN.md)

# TSCM-Calib

The multi-camera calibration tool based on Triple Sphere Camera Model.

## Dependence

* OpenCV

  you can install it by `sudo apt install libopencv-dev`.

* Ceres-Solver

  you can install it by `sudo apt install libceres-dev`.

## build

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Change the image path in `main` function. The tool will output the calibration result in yaml format, saved in `build` directory.

> Note: press `Esc` or `Q` to exit.

> Note: The filenames should be exactly same for the pictures captured simulately from different cameras.

> Note: The Triple Sphere Camera Model can be found in [OmniVidar](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_OmniVidar_Omnidirectional_Depth_Estimation_From_Multi-Fisheye_Images_CVPR_2023_paper.pdf).
