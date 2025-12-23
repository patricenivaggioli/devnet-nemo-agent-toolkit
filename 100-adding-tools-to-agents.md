# Adding Tools To Agents

In this lab, we showcase how the NVIDIA NeMo Agent Toolkit (NAT) allows developers to easily add tool-calling capabilities to agents that integrate with the library. Tool calling allows chain-of-thought planners to augment the pretrained capabilities of LLMs with predefined methods and access to context retrieval services. This is a powerful paradigm in agentic AI development that has enabled revolutionary technologies like deep research and API-integrated AI systems.

## Data Sources

Several data files are required for this example. To keep this as a stand-alone example, the files are included here as cells which can be run to create them.

The following commands creates the `data` directory as well as a `rag` subdirectory.

```bash
cd ~/work/nemo-agent-toolkit-clone/
mkdir -p data/rag
```

The following cell writes the `data/retail_sales_data.csv` file.

```bash
cat > data/retail_sales_data.csv <<'EOF'
Date,StoreID,Product,UnitsSold,Revenue,Promotion
2024-01-01,S001,Laptop,1,1000,No
2024-01-01,S001,Phone,9,4500,No
2024-01-01,S001,Tablet,2,600,No
2024-01-01,S002,Laptop,9,9000,No
2024-01-01,S002,Phone,10,5000,No
2024-01-01,S002,Tablet,5,1500,No
2024-01-02,S001,Laptop,4,4000,No
2024-01-02,S001,Phone,11,5500,No
2024-01-02,S001,Tablet,7,2100,No
2024-01-02,S002,Laptop,7,7000,No
2024-01-02,S002,Phone,6,3000,No
2024-01-02,S002,Tablet,9,2700,No
2024-01-03,S001,Laptop,6,6000,No
2024-01-03,S001,Phone,7,3500,No
2024-01-03,S001,Tablet,8,2400,No
2024-01-03,S002,Laptop,3,3000,No
2024-01-03,S002,Phone,16,8000,No
2024-01-03,S002,Tablet,5,1500,No
2024-01-04,S001,Laptop,5,5000,No
2024-01-04,S001,Phone,11,5500,No
2024-01-04,S001,Tablet,9,2700,No
2024-01-04,S002,Laptop,2,2000,No
2024-01-04,S002,Phone,12,6000,No
2024-01-04,S002,Tablet,7,2100,No
2024-01-05,S001,Laptop,8,8000,No
2024-01-05,S001,Phone,18,9000,No
2024-01-05,S001,Tablet,5,1500,No
2024-01-05,S002,Laptop,7,7000,No
2024-01-05,S002,Phone,10,5000,No
2024-01-05,S002,Tablet,10,3000,No
2024-01-06,S001,Laptop,9,9000,No
2024-01-06,S001,Phone,11,5500,No
2024-01-06,S001,Tablet,5,1500,No
2024-01-06,S002,Laptop,5,5000,No
2024-01-06,S002,Phone,14,7000,No
2024-01-06,S002,Tablet,10,3000,No
2024-01-07,S001,Laptop,2,2000,No
2024-01-07,S001,Phone,15,7500,No
2024-01-07,S001,Tablet,6,1800,No
2024-01-07,S002,Laptop,0,0,No
2024-01-07,S002,Phone,7,3500,No
2024-01-07,S002,Tablet,12,3600,No
2024-01-08,S001,Laptop,5,5000,No
2024-01-08,S001,Phone,8,4000,No
2024-01-08,S001,Tablet,5,1500,No
2024-01-08,S002,Laptop,4,4000,No
2024-01-08,S002,Phone,11,5500,No
2024-01-08,S002,Tablet,9,2700,No
2024-01-09,S001,Laptop,6,6000,No
2024-01-09,S001,Phone,9,4500,No
2024-01-09,S001,Tablet,8,2400,No
2024-01-09,S002,Laptop,7,7000,No
2024-01-09,S002,Phone,11,5500,No
2024-01-09,S002,Tablet,8,2400,No
2024-01-10,S001,Laptop,6,6000,No
2024-01-10,S001,Phone,11,5500,No
2024-01-10,S001,Tablet,5,1500,No
2024-01-10,S002,Laptop,8,8000,No
2024-01-10,S002,Phone,5,2500,No
2024-01-10,S002,Tablet,6,1800,No
2024-01-11,S001,Laptop,5,5000,No
2024-01-11,S001,Phone,7,3500,No
2024-01-11,S001,Tablet,5,1500,No
2024-01-11,S002,Laptop,4,4000,No
2024-01-11,S002,Phone,10,5000,No
2024-01-11,S002,Tablet,4,1200,No
2024-01-12,S001,Laptop,2,2000,No
2024-01-12,S001,Phone,10,5000,No
2024-01-12,S001,Tablet,9,2700,No
2024-01-12,S002,Laptop,8,8000,No
2024-01-12,S002,Phone,10,5000,No
2024-01-12,S002,Tablet,14,4200,No
2024-01-13,S001,Laptop,3,3000,No
2024-01-13,S001,Phone,6,3000,No
2024-01-13,S001,Tablet,9,2700,No
2024-01-13,S002,Laptop,1,1000,No
2024-01-13,S002,Phone,12,6000,No
2024-01-13,S002,Tablet,7,2100,No
2024-01-14,S001,Laptop,4,4000,Yes
2024-01-14,S001,Phone,16,8000,Yes
2024-01-14,S001,Tablet,4,1200,Yes
2024-01-14,S002,Laptop,5,5000,Yes
2024-01-14,S002,Phone,14,7000,Yes
2024-01-14,S002,Tablet,6,1800,Yes
2024-01-15,S001,Laptop,9,9000,No
2024-01-15,S001,Phone,6,3000,No
2024-01-15,S001,Tablet,11,3300,No
2024-01-15,S002,Laptop,5,5000,No
2024-01-15,S002,Phone,10,5000,No
2024-01-15,S002,Tablet,4,1200,No
2024-01-16,S001,Laptop,6,6000,No
2024-01-16,S001,Phone,11,5500,No
2024-01-16,S001,Tablet,5,1500,No
2024-01-16,S002,Laptop,4,4000,No
2024-01-16,S002,Phone,7,3500,No
2024-01-16,S002,Tablet,4,1200,No
2024-01-17,S001,Laptop,6,6000,No
2024-01-17,S001,Phone,14,7000,No
2024-01-17,S001,Tablet,7,2100,No
2024-01-17,S002,Laptop,3,3000,No
2024-01-17,S002,Phone,7,3500,No
2024-01-17,S002,Tablet,6,1800,No
2024-01-18,S001,Laptop,7,7000,Yes
2024-01-18,S001,Phone,10,5000,Yes
2024-01-18,S001,Tablet,6,1800,Yes
2024-01-18,S002,Laptop,5,5000,Yes
2024-01-18,S002,Phone,16,8000,Yes
2024-01-18,S002,Tablet,8,2400,Yes
2024-01-19,S001,Laptop,4,4000,No
2024-01-19,S001,Phone,12,6000,No
2024-01-19,S001,Tablet,7,2100,No
2024-01-19,S002,Laptop,3,3000,No
2024-01-19,S002,Phone,12,6000,No
2024-01-19,S002,Tablet,8,2400,No
2024-01-20,S001,Laptop,6,6000,No
2024-01-20,S001,Phone,8,4000,No
2024-01-20,S001,Tablet,6,1800,No
2024-01-20,S002,Laptop,8,8000,No
2024-01-20,S002,Phone,9,4500,No
2024-01-20,S002,Tablet,8,2400,No
2024-01-21,S001,Laptop,3,3000,No
2024-01-21,S001,Phone,9,4500,No
2024-01-21,S001,Tablet,5,1500,No
2024-01-21,S002,Laptop,8,8000,No
2024-01-21,S002,Phone,15,7500,No
2024-01-21,S002,Tablet,7,2100,No
2024-01-22,S001,Laptop,1,1000,No
2024-01-22,S001,Phone,15,7500,No
2024-01-22,S001,Tablet,5,1500,No
2024-01-22,S002,Laptop,11,11000,No
2024-01-22,S002,Phone,4,2000,No
2024-01-22,S002,Tablet,4,1200,No
2024-01-23,S001,Laptop,3,3000,No
2024-01-23,S001,Phone,8,4000,No
2024-01-23,S001,Tablet,8,2400,No
2024-01-23,S002,Laptop,6,6000,No
2024-01-23,S002,Phone,12,6000,No
2024-01-23,S002,Tablet,12,3600,No
2024-01-24,S001,Laptop,2,2000,No
2024-01-24,S001,Phone,14,7000,No
2024-01-24,S001,Tablet,6,1800,No
2024-01-24,S002,Laptop,1,1000,No
2024-01-24,S002,Phone,5,2500,No
2024-01-24,S002,Tablet,7,2100,No
2024-01-25,S001,Laptop,7,7000,No
2024-01-25,S001,Phone,11,5500,No
2024-01-25,S001,Tablet,11,3300,No
2024-01-25,S002,Laptop,6,6000,No
2024-01-25,S002,Phone,11,5500,No
2024-01-25,S002,Tablet,5,1500,No
2024-01-26,S001,Laptop,5,5000,Yes
2024-01-26,S001,Phone,22,11000,Yes
2024-01-26,S001,Tablet,7,2100,Yes
2024-01-26,S002,Laptop,6,6000,Yes
2024-01-26,S002,Phone,24,12000,Yes
2024-01-26,S002,Tablet,3,900,Yes
2024-01-27,S001,Laptop,7,7000,Yes
2024-01-27,S001,Phone,20,10000,Yes
2024-01-27,S001,Tablet,6,1800,Yes
2024-01-27,S002,Laptop,4,4000,Yes
2024-01-27,S002,Phone,8,4000,Yes
2024-01-27,S002,Tablet,6,1800,Yes
2024-01-28,S001,Laptop,10,10000,No
2024-01-28,S001,Phone,15,7500,No
2024-01-28,S001,Tablet,12,3600,No
2024-01-28,S002,Laptop,6,6000,No
2024-01-28,S002,Phone,11,5500,No
2024-01-28,S002,Tablet,10,3000,No
2024-01-29,S001,Laptop,3,3000,No
2024-01-29,S001,Phone,16,8000,No
2024-01-29,S001,Tablet,5,1500,No
2024-01-29,S002,Laptop,6,6000,No
2024-01-29,S002,Phone,17,8500,No
2024-01-29,S002,Tablet,2,600,No
2024-01-30,S001,Laptop,3,3000,No
2024-01-30,S001,Phone,11,5500,No
2024-01-30,S001,Tablet,2,600,No
2024-01-30,S002,Laptop,6,6000,No
2024-01-30,S002,Phone,16,8000,No
2024-01-30,S002,Tablet,8,2400,No
2024-01-31,S001,Laptop,5,5000,Yes
2024-01-31,S001,Phone,22,11000,Yes
2024-01-31,S001,Tablet,9,2700,Yes
2024-01-31,S002,Laptop,3,3000,Yes
2024-01-31,S002,Phone,14,7000,Yes
2024-01-31,S002,Tablet,4,1200,Yes
2024-02-01,S001,Laptop,2,2000,No
2024-02-01,S001,Phone,7,3500,No
2024-02-01,S001,Tablet,11,3300,No
2024-02-01,S002,Laptop,6,6000,No
2024-02-01,S002,Phone,11,5500,No
2024-02-01,S002,Tablet,5,1500,No
2024-02-02,S001,Laptop,2,2000,No
2024-02-02,S001,Phone,9,4500,No
2024-02-02,S001,Tablet,7,2100,No
2024-02-02,S002,Laptop,5,5000,No
2024-02-02,S002,Phone,9,4500,No
2024-02-02,S002,Tablet,12,3600,No
2024-02-03,S001,Laptop,9,9000,No
2024-02-03,S001,Phone,12,6000,No
2024-02-03,S001,Tablet,9,2700,No
2024-02-03,S002,Laptop,10,10000,No
2024-02-03,S002,Phone,6,3000,No
2024-02-03,S002,Tablet,10,3000,No
2024-02-04,S001,Laptop,6,6000,No
2024-02-04,S001,Phone,5,2500,No
2024-02-04,S001,Tablet,8,2400,No
2024-02-04,S002,Laptop,6,6000,No
2024-02-04,S002,Phone,10,5000,No
2024-02-04,S002,Tablet,10,3000,No
2024-02-05,S001,Laptop,7,7000,No
2024-02-05,S001,Phone,13,6500,No
2024-02-05,S001,Tablet,11,3300,No
2024-02-05,S002,Laptop,8,8000,No
2024-02-05,S002,Phone,11,5500,No
2024-02-05,S002,Tablet,8,2400,No
2024-02-06,S001,Laptop,5,5000,No
2024-02-06,S001,Phone,14,7000,No
2024-02-06,S001,Tablet,4,1200,No
2024-02-06,S002,Laptop,2,2000,No
2024-02-06,S002,Phone,11,5500,No
2024-02-06,S002,Tablet,7,2100,No
2024-02-07,S001,Laptop,6,6000,No
2024-02-07,S001,Phone,7,3500,No
2024-02-07,S001,Tablet,9,2700,No
2024-02-07,S002,Laptop,2,2000,No
2024-02-07,S002,Phone,8,4000,No
2024-02-07,S002,Tablet,9,2700,No
2024-02-08,S001,Laptop,5,5000,No
2024-02-08,S001,Phone,12,6000,No
2024-02-08,S001,Tablet,3,900,No
2024-02-08,S002,Laptop,8,8000,No
2024-02-08,S002,Phone,5,2500,No
2024-02-08,S002,Tablet,8,2400,No
2024-02-09,S001,Laptop,6,6000,Yes
2024-02-09,S001,Phone,18,9000,Yes
2024-02-09,S001,Tablet,5,1500,Yes
2024-02-09,S002,Laptop,7,7000,Yes
2024-02-09,S002,Phone,18,9000,Yes
2024-02-09,S002,Tablet,5,1500,Yes
2024-02-10,S001,Laptop,9,9000,No
2024-02-10,S001,Phone,6,3000,No
2024-02-10,S001,Tablet,8,2400,No
2024-02-10,S002,Laptop,7,7000,No
2024-02-10,S002,Phone,5,2500,No
2024-02-10,S002,Tablet,6,1800,No
2024-02-11,S001,Laptop,6,6000,No
2024-02-11,S001,Phone,11,5500,No
2024-02-11,S001,Tablet,2,600,No
2024-02-11,S002,Laptop,7,7000,No
2024-02-11,S002,Phone,5,2500,No
2024-02-11,S002,Tablet,9,2700,No
2024-02-12,S001,Laptop,5,5000,No
2024-02-12,S001,Phone,5,2500,No
2024-02-12,S001,Tablet,4,1200,No
2024-02-12,S002,Laptop,1,1000,No
2024-02-12,S002,Phone,14,7000,No
2024-02-12,S002,Tablet,15,4500,No
2024-02-13,S001,Laptop,3,3000,No
2024-02-13,S001,Phone,18,9000,No
2024-02-13,S001,Tablet,8,2400,No
2024-02-13,S002,Laptop,5,5000,No
2024-02-13,S002,Phone,8,4000,No
2024-02-13,S002,Tablet,6,1800,No
2024-02-14,S001,Laptop,4,4000,No
2024-02-14,S001,Phone,9,4500,No
2024-02-14,S001,Tablet,6,1800,No
2024-02-14,S002,Laptop,4,4000,No
2024-02-14,S002,Phone,6,3000,No
2024-02-14,S002,Tablet,7,2100,No
2024-02-15,S001,Laptop,4,4000,Yes
2024-02-15,S001,Phone,26,13000,Yes
2024-02-15,S001,Tablet,5,1500,Yes
2024-02-15,S002,Laptop,2,2000,Yes
2024-02-15,S002,Phone,14,7000,Yes
2024-02-15,S002,Tablet,6,1800,Yes
2024-02-16,S001,Laptop,7,7000,No
2024-02-16,S001,Phone,9,4500,No
2024-02-16,S001,Tablet,1,300,No
2024-02-16,S002,Laptop,6,6000,No
2024-02-16,S002,Phone,12,6000,No
2024-02-16,S002,Tablet,10,3000,No
2024-02-17,S001,Laptop,5,5000,No
2024-02-17,S001,Phone,8,4000,No
2024-02-17,S001,Tablet,14,4200,No
2024-02-17,S002,Laptop,4,4000,No
2024-02-17,S002,Phone,13,6500,No
2024-02-17,S002,Tablet,7,2100,No
2024-02-18,S001,Laptop,6,6000,Yes
2024-02-18,S001,Phone,22,11000,Yes
2024-02-18,S001,Tablet,9,2700,Yes
2024-02-18,S002,Laptop,2,2000,Yes
2024-02-18,S002,Phone,10,5000,Yes
2024-02-18,S002,Tablet,12,3600,Yes
2024-02-19,S001,Laptop,6,6000,No
2024-02-19,S001,Phone,12,6000,No
2024-02-19,S001,Tablet,3,900,No
2024-02-19,S002,Laptop,3,3000,No
2024-02-19,S002,Phone,4,2000,No
2024-02-19,S002,Tablet,7,2100,No
EOF
```

The following cell writes the RAG product catalog file, `data/rag/product_catalog.md`.

```bash
cat > data/rag/product_catalog.md <<'EOF'
# Product Catalog: Smartphones, Laptops, and Tablets

## Smartphones

The Veltrix Solis Z9 is a flagship device in the premium smartphone segment. It builds on a decade of design iterations that prioritize screen-to-body ratio, minimal bezels, and high refresh rate displays. The 6.7-inch AMOLED panel with 120Hz refresh rate delivers immersive visual experiences, whether in gaming, video streaming, or augmented reality applications. The display's GorillaGlass Fusion coating provides scratch resistance and durability, and the thin form factor is engineered using a titanium-aluminum alloy chassis to reduce weight without compromising rigidity.

Internally, the Solis Z9 is powered by the OrionEdge V14 chipset, a 4nm process SoC designed for high-efficiency workloads. Its AI accelerator module handles on-device tasks such as voice transcription, camera optimization, and intelligent background app management. The inclusion of 12GB LPDDR5 RAM and a 256GB UFS 3.1 storage system allows for seamless multitasking, instant app launching, and rapid data access. The device supports eSIM and dual physical SIM configurations, catering to global travelers and hybrid network users.

Photography and videography are central to the Solis Z9 experience. The triple-camera system incorporates a periscope-style 8MP telephoto lens with 5x optical zoom, a 12MP ultra-wide sensor with macro capabilities, and a 64MP main sensor featuring optical image stabilization (OIS) and phase detection autofocus (PDAF). Night mode and HDRX+ processing enable high-fidelity image capture in challenging lighting conditions.

Software-wise, the device ships with LunOS 15, a lightweight Android fork optimized for modular updates and privacy compliance. The system supports secure containers for work profiles and AI-powered notifications that summarize app alerts across channels. Facial unlock is augmented by a 3D IR depth sensor, providing reliable biometric security alongside the ultrasonic in-display fingerprint scanner.

The Solis Z9 is a culmination of over a decade of design experimentation in mobile form factors, ranging from curved-edge screens to under-display camera arrays. Its balance of performance, battery efficiency, and user-centric software makes it an ideal daily driver for content creators, mobile gamers, and enterprise users.

## Laptops

The Cryon Vanta 16X represents the latest evolution of portable computing power tailored for professional-grade workloads.

The Vanta 16X features a unibody chassis milled from aircraft-grade aluminum using CNC machining. The thermal design integrates vapor chamber cooling and dual-fan exhaust architecture to support sustained performance under high computational loads. The 16-inch 4K UHD display is color-calibrated at the factory and supports HDR10+, making it suitable for cinematic video editing and high-fidelity CAD modeling.

Powering the device is Intel's Core i9-13900H processor, which includes 14 cores with a hybrid architecture combining performance and efficiency cores. This allows the system to dynamically balance power consumption and raw speed based on active workloads. The dedicated Zephira RTX 4700G GPU features 8GB of GDDR6 VRAM and is optimized for CUDA and Tensor Core operations, enabling applications in real-time ray tracing, AI inference, and 3D rendering.

The Vanta 16X includes a 2TB PCIe Gen 4 NVMe SSD, delivering sequential read/write speeds above 7GB/s, and 32GB of high-bandwidth DDR5 RAM. The machine supports hardware-accelerated virtualization and dual-booting, and ships with VireoOS Pro pre-installed, with official drivers available for Fedora, Ubuntu LTS, and NebulaOS.

Input options are expansive. The keyboard features per-key RGB lighting and programmable macros, while the haptic touchpad supports multi-gesture navigation and palm rejection. Port variety includes dual Thunderbolt 4 ports, a full-size SD Express card reader, HDMI 2.1, 2.5G Ethernet, three USB-A 3.2 ports, and a 3.5mm TRRS audio jack. A fingerprint reader is embedded in the power button and supports biometric logins via Windows Hello.

The history of the Cryon laptop line dates back to the early 2010s, when the company launched its first ultrabook aimed at mobile developers. Since then, successive generations have introduced carbon fiber lids, modular SSD bays, and convertible form factors. The Vanta 16X continues this tradition by integrating a customizable BIOS, a modular fan assembly, and a trackpad optimized for creative software like Blender and Adobe Creative Suite.

Designed for software engineers, data scientists, film editors, and 3D artists, the Cryon Vanta 16X is a workstation-class laptop in a portable shell.

## Tablets

The Nebulyn Ark S12 Ultra reflects the current apex of tablet technology, combining high-end hardware with software environments tailored for productivity and creativity.

The Ark S12 Ultra is built around a 12.9-inch OLED display that supports 144Hz refresh rate and HDR10+ dynamic range. With a resolution of 2800 x 1752 pixels and a contrast ratio of 1,000,000:1, the screen delivers vibrant color reproduction ideal for design and media consumption. The display supports true tone adaptation and low blue-light filtering for prolonged use.

Internally, the tablet uses Qualcomm's Snapdragon 8 Gen 3 SoC, which includes an Adreno 750 GPU and an NPU for on-device AI tasks. The device ships with 16GB LPDDR5X RAM and 512GB of storage with support for NVMe expansion via a proprietary magnetic dock. The 11200mAh battery enables up to 15 hours of typical use and recharges to 80 percent in 45 minutes via 45W USB-C PD.

The Ark's history traces back to the original Nebulyn Tab, which launched in 2014 as an e-reader and video streaming device. Since then, the line has evolved through multiple iterations that introduced stylus support, high-refresh screens, and multi-window desktop modes. The current model supports NebulynVerse, a DeX-like environment that allows external display mirroring and full multitasking with overlapping windows and keyboard shortcuts.

Input capabilities are central to the Ark S12 Ultra’s appeal. The Pluma Stylus 3 features magnetic charging, 4096 pressure levels, and tilt detection. It integrates haptic feedback to simulate traditional pen strokes and brush textures. The device also supports a SnapCover keyboard that includes a trackpad and programmable shortcut keys. With the stylus and keyboard, users can effectively transform the tablet into a mobile workstation or digital sketchbook.

Camera hardware includes a 13MP main sensor and a 12MP ultra-wide front camera with center-stage tracking and biometric unlock. Microphone arrays with beamforming enable studio-quality call audio. Connectivity includes Wi-Fi 7, Bluetooth 5.3, and optional LTE/5G with eSIM.

Software support is robust. The device runs NebulynOS 6.0, based on Android 14L, and supports app sandboxing, multi-user profiles, and remote device management. Integration with cloud services, including SketchNimbus and ThoughtSpace, allows for real-time collaboration and syncing of content across devices.

This tablet is targeted at professionals who require a balance between media consumption, creativity, and light productivity. Typical users include architects, consultants, university students, and UX designers.

## Comparative Summary

Each of these devices—the Veltrix Solis Z9, Cryon Vanta 16X, and Nebulyn Ark S12 Ultra—represents a best-in-class interpretation of its category. The Solis Z9 excels in mobile photography and everyday communication. The Vanta 16X is tailored for high-performance applications such as video production and AI prototyping. The Ark S12 Ultra provides a canvas for creativity, note-taking, and hybrid productivity use cases.

## Historical Trends and Design Evolution

Design across all three categories is converging toward modularity, longevity, and environmental sustainability. Recycled materials, reparability scores, and software longevity are becoming integral to brand reputation and product longevity. Future iterations are expected to feature tighter integration with wearable devices, ambient AI experiences, and cross-device workflows.
EOF
```

## Llama-index subpackage

As for the previous lab, NeMo Agent toolkit can be installed through the PyPI `nvidia-nat` package.

There are several optional subpackages available for NAT. For this example, we will rely on two subpackages:

* The `langchain` subpackage contains useful components for integrating and running within [LangChain](https://python.langchain.com/docs/introduction/).
* The `llama-index` subpackage contains useful components for integrating and running within [LlamaIndex](https://developers.llamaindex.ai/python/framework/).

As `langchain` subpackage is already installed, let's install `llama-index`:

```bash
uv pip install "nvidia-nat[llama-index]"
```

## Creating a New Workflow

As explained in detail in previous lab, we can use the `nat workflow create` sub-command to create the necessary directory structure for a new agent.

Within this directory we can define all of the functions that we want to be available to the agent at runtime. In this notebook specifically we are going to demonstrate the integration of new tools to the workflow. We will make them available or 'discoverable' by the agent by defining these tool calls within the `register.py` function.

```bash
cd ~/work/nemo-agent-toolkit-clone/
nat workflow create retail_sales_agent
```

## Defining New Tools

Next we will show you how to add new tools to the agent.

### Total Product Sales Data Tool

This tool gets total sales for a specific product from `data/product_catalog.md`

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/src/retail_sales_agent/total_product_sales_data_tool.py <<'EOF'
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GetTotalProductSalesDataConfig(FunctionBaseConfig, name="get_total_product_sales_data"):
    """Get total sales data by product."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=GetTotalProductSalesDataConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_total_product_sales_data_function(config: GetTotalProductSalesDataConfig, _builder: Builder):
    """Get total sales data for a specific product."""
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _get_total_product_sales_data(product_name: str) -> str:
        """
        Retrieve total sales data for a specific product.

        Args:
            product_name: Name of the product

        Returns:
            String message containing total sales data
        """
        df['Product'] = df["Product"].apply(lambda x: x.lower())
        revenue = df[df['Product'] == product_name]['Revenue'].sum()
        units_sold = df[df['Product'] == product_name]['UnitsSold'].sum()

        return f"Revenue for {product_name} are {revenue} and total units sold are {units_sold}"

    yield FunctionInfo.from_fn(
        _get_total_product_sales_data,
        description=_get_total_product_sales_data.__doc__)
EOF
```

### Sales Per Day Tool

This tool gets the total sales across all products per day from `data/product_catalog.md`

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/src/retail_sales_agent/sales_per_day_tool.py <<'EOF'
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GetSalesPerDayConfig(FunctionBaseConfig, name="get_sales_per_day"):
    """Get total sales across all products per day."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=GetSalesPerDayConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def sales_per_day_function(config: GetSalesPerDayConfig, builder: Builder):
    """Get total sales across all products per day."""
    import pandas as pd

    df = pd.read_csv(config.data_path)
    df['Product'] = df["Product"].apply(lambda x: x.lower())

    async def _get_sales_per_day(date: str, product: str) -> str:
        """
        Calculate total sales data across all products for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            product: Product name

        Returns:
            String message with the total sales for the day
        """
        if date == "None":
            return "Please provide a date in YYYY-MM-DD format."
        total_revenue = df[(df['Date'] == date) & (df['Product'] == product)]['Revenue'].sum()
        total_units_sold = df[(df['Date'] == date) & (df['Product'] == product)]['UnitsSold'].sum()

        return f"Total revenue for {date} is {total_revenue} and total units sold is {total_units_sold}"

    yield FunctionInfo.from_fn(
        _get_sales_per_day,
        description=_get_sales_per_day.__doc__)
EOF
```

### Detect Outliers Tool

This tool detects outliers in `data/product_catalog.md` data using IQR (Interquartile Range) method.

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/src/retail_sales_agent/detect_outliers_tool.py <<'EOF'
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class DetectOutliersIQRConfig(FunctionBaseConfig, name="detect_outliers_iqr"):
    """Detect outliers in sales data using IQR method."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=DetectOutliersIQRConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def detect_outliers_iqr_function(config: DetectOutliersIQRConfig, _builder: Builder):
    """Detect outliers in sales data using the Interquartile Range (IQR) method."""
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _detect_outliers_iqr(metric: str) -> str:
        """
        Detect outliers in retail data using the IQR method.

        Args:
            metric: Specific metric to check for outliers

        Returns:
            Dictionary containing outlier analysis results
        """
        if metric == "None":
            column = "Revenue"
        else:
            column = metric

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)]

        return f"Outliers in {column} are {outliers.to_dict('records')}"

    yield FunctionInfo.from_fn(
        _detect_outliers_iqr,
        description=_detect_outliers_iqr.__doc__)
EOF
```

## Registering Tools

We need to update the `register.py` file to register these tools with NeMo Agent toolkit.

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/src/retail_sales_agent/register.py <<'EOF'

from . import sales_per_day_tool
from . import detect_outliers_tool
from . import total_product_sales_data_tool
EOF
```

## Updating The Configuration File

Below we show how to update the default configuration file for this new tool-calling retail sales agent with the new tools (python methods) that have been defined and registered properly.

Take a moment to analyze the new `retail_sales_agent/configs/config.yml` file below, where a `functions` header has been defined and each registered tool from the previous section is listed. 

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/configs/config.yml <<'EOF'
llms:
  azure_llm:
    _type: azure_openai
    azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
    azure_deployment: ${AZURE_OPENAI_DEPLOYMENT}
    api_key: ${AZURE_OPENAI_API_KEY}
    api_version: ${AZURE_OPENAI_API_VERSION}

functions:
  total_product_sales_data:
    _type: get_total_product_sales_data
    data_path: data/retail_sales_data.csv
  sales_per_day:
    _type: get_sales_per_day
    data_path: data/retail_sales_data.csv
  detect_outliers:
    _type: detect_outliers_iqr
    data_path: data/retail_sales_data.csv

workflow:
  _type: react_agent
  tool_names:
    - total_product_sales_data
    - sales_per_day
    - detect_outliers
  llm_name: azure_llm
  verbose: true
  handle_parsing_errors: true
  max_retries: 2
  description: "A helpful assistant that can answer questions about the retail sales CSV data"
EOF
```

## Running the Initial Workflow

The workflow has been properly created, new tools defined, registered, and incorporated into the config.yml. We are now ready to run the agent and test out that the ReAct agent is able to properly determine the intent of the input query and complete the necessary tool calling to serve the user.

```bash
nat workflow reinstall retail_sales_agent
```

This first query asks how laptop sales compare to phone sales.

In the output, we expect to see calls to the `total_product_sales_data` tool.

```bash
nat run --config_file=retail_sales_agent/configs/config.yml --input "How do laptop sales compare to phone sales?"
```

In this next query we ask what were the laptop sales on a specific date.

In the output, we expect to see a call to the `sales_per_day` tool.

```bash
nat run --config_file=retail_sales_agent/configs/config.yml --input "What were the laptop sales on February 16th 2024?"
```

In the last query we ask if there were any outliers in sales.

In the output, we expect to see a call to the `detect_outliers` tool.

```bash
nat run --config_file=retail_sales_agent/configs/config.yml --input "What were the outliers in 'Revenue'?"
```

