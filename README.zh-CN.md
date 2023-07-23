[English](README.md) | 中文简体

# TSCM-Calib

基于`Triple Sphere Camera Model`的多相机标定工具。

## Dependence

* OpenCV

  在ubuntu下安装： `sudo apt install libopencv-dev`.

* Ceres-Solver

  在ubuntu下安装： `sudo apt install libceres-dev`.

## build

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

修改`main`函数里的图片路径，可根据自己的需求增删相机数。标定结束后会在`build`目录下生成`yaml`格式的标定文件。

`EpipolarRecity`文件夹下是用于四目鱼眼相机极线校正的小工具，对应于了论文[OmniVidar](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_OmniVidar_Omnidirectional_Depth_Estimation_From_Multi-Fisheye_Images_CVPR_2023_paper.pdf)的3.2小节。

> Note: 标定结束后会出现一个多相机位姿可视化窗口，按下键盘上的`Esc`或`Q`键，或者直接关闭该窗口可以退出程序。

> Note: 不同相机在同一时刻拍摄的照片的文件名应当保持一致。

> Note: 相机模型的投影方程和逆投影方程可以参考这篇论文： [OmniVidar](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_OmniVidar_Omnidirectional_Depth_Estimation_From_Multi-Fisheye_Images_CVPR_2023_paper.pdf).
