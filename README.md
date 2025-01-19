# TacFlex: Multi-Mode Tactile Imprints Simulation for Visuotactile Sensors with Coating Patterns

![图片](https://github.com/user-attachments/assets/bea5970c-ee31-40cb-ab54-5f8ad9614220)


## Installation

**Requirements:**

- Python 3.8.x-3.10.x
- NVIDIA Isaac Gym preview 4 release (download from https://developer.nvidia.com/isaac-gym/download)
- SOFA framework 24.06 (download from https://www.sofa-framework.org/download/)
- pyrender 0.1.45
- trimesh 3.23.5
- scikit-learn 1.3.0



Clone this repo with

```bash
git clone https://github.com/Zhang-Chaofan/TacFlex.git
```

## Quick-start: demo

To simulate the GelSight Mini sensor with marker, run the following:
```bash
python examples/sim_gelsight_mini.py --indenter=cross --render_img
```

To simulate the GelStereo 2.0 sensor with a marker pattern, run the following:
```bash
python examples/sim_gelstereo.py --pattern=marker --indenter=triangle_6-2 --sim_pcd --render_img
```

To simulate the GelStereo 2.0 sensor with a checkerboard pattern, run the following:
```bash
python examples/sim_gelstereo.py --pattern=checker --indenter=star --render_img
```

To simulate the GelStereo 2.0 sensor with a random color pattern, run the following:
```bash
python examples/sim_gelstereo.py --pattern=random --indenter=hexagon_4-3 --render_img
```

To simulate the soft-bubble tactile sensor (require SOFA framework), run the following:
```bash
python examples/sim_soft_bubble.py
python examples/sim_soft_bubble_2.py  # rgb image with refraction effect
```



## Tutorial for simulating custom sensors

We are working on the detailed tutorial.


