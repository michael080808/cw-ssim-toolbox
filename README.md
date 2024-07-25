# 复小波结构相似性工具箱<br>Toolbox for Complex Wavelet<br>Structural Similarity Index Measure (CW-SSIM)

* 本项目是应用以下信号分析方法实现的CW-SSIM算法
* A little toolbox for two kinds of implementations of CW-SSIM with
    * 复数可控金字塔
    * Complex Steerable Pyramid
    * 双树复小波变换
    * Dual Tree-Complex Wavelet Transform (DT-CWT)


* 特别感谢[Nick Kingsbury教授](http://www-sigproc.eng.cam.ac.uk/Main/NGK)的工作以及[dtcwt_matlab](https://github.com/timseries/dtcwt_matlab)项目分享的Matlab代码
* Special Thanks for [Prof. Nick Kingsbury](http://www-sigproc.eng.cam.ac.uk/Main/NGK) and the code from [dtcwt_matlab](https://github.com/timseries/dtcwt_matlab) repository.


* 声明：受限于作者知识水平，本项目可能存在运算错误
* Notice: There may be some errors in project due to author's limited knowledge.


* 如发现计算问题，欢迎提交中文或英文的Issue
* Glad to receive Chinese/English Issue if anyone finds calculation problems.

## 项目特色（Highlights）

* 实现了任意维度数据的双树复小波变换（1维、2维、3维、N维）
* Apply Dual Tree Complex Wavelet Transform (DT-CWT) on data with any dimensions (1-D, 2-D, 3-D, n-D).


* 输入数据可以高于双树复小波变换的实际处理维度
* Input dimension can be higher than the DT-CWT processing dimension


* 基于[dtcwt_matlab](https://github.com/timseries/dtcwt_matlab)中`wavegen.m`小波生成代码，实现了小波的按需求精度生成
* According to `wavegen.m` from [dtcwt_matlab](https://github.com/timseries/dtcwt_matlab)，wavelet can generate as demand.


* 实现了小波数值的缓存和数值保存，方便快速调用
* Support caching and storing values of wavelets to speed up calculation.


* CPU运算仅使用了`numpy`/`scipy`/`sympy`
* Only use `numpy`/`scipy`/`sympy` to achieve CPU calculation.


* GPU加速支持`cupy`和`torch`/`pytorch`
* Support GPU acceleration with `cupy` and `torch`/`pytorch`


* 各部分模块化，方便导入，易于扩展
* All the module are independent with each other. Easy to import and extent.

## 代码效果（Visuals)

## 使用方法（Usage)

## 特殊说明（Notes)

* 由于`cupy`和`torch`/`pytorch`在下列函数的支持方面存在问题，小波计算使用`numpy`
* All wavelets calculate with `numpy` due to the compatible issues with `cupy` and `torch`/`pytorch` functions.
    * `cupy`
        * [`cupy.roots`](https://docs.cupy.dev/en/stable/reference/generated/cupy.roots.html) 和 [`cupy.poly`](https://docs.cupy.dev/en/stable/reference/generated/cupy.poly.html) 多项式系数不支持伴随矩阵为任意矩阵的特征值求解
        * [`cupy.roots`](https://docs.cupy.dev/en/stable/reference/generated/cupy.roots.html) and [`cupy.poly`](https://docs.cupy.dev/en/stable/reference/generated/cupy.poly.html) functions do not support polynomial coefficients whose companion matrices are general 2d square arrays.
    * `torch`/`pytorch`
        * `torch`/`pytorch`中没有`numpy.roots`, `numpy.poly`, `scipy.linalg.toeplitz`的直接支持
        * `torch`/`pytorch` is lack of direct supports with `numpy.roots`, `numpy.poly` and `scipy.linalg.toeplitz`.