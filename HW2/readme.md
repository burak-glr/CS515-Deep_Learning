# Transfer learning and Knowledge Distilation

Entry point for HW2: Transfer Learning and Knowledge Distillation.

Dispatches to one of three task pipelines based on ``--task``:

  classify  — train/test a model (CNN, ResNet, MobileNet) from scratch,
               with optional label smoothing.
  transfer  — fine-tune a pretrained ResNet-18 or VGG-16 on CIFAR-10
               using Option 1 (resize + freeze) or Option 2 (small conv + fine-tune).
  distill   — knowledge distillation from a saved ResNet teacher into a
               smaller student (CNN or MobileNet); supports standard KD and
               a modified variant where the teacher probability is assigned
               to the true class only.

Usage examples::

    # Train ResNet-18 from scratch (no label smoothing)
    python main.py --task classify --model resnet --epochs 50 --device cuda

    # Train ResNet-18 from scratch WITH label smoothing (ε=0.1)
    python main.py --task classify --model resnet --label_smoothing 0.1 --epochs 50

    # Transfer learning – Option 1
    python main.py --task transfer --arch resnet18 --transfer_option 1 --epochs 10

    # Transfer learning – Option 2
    python main.py --task transfer --arch resnet18 --transfer_option 2 --epochs 20

    # Standard KD: teacher → SimpleCNN
    python main.py --task distill --model cnn --distill_mode standard \\
                   --teacher_path best_teacher.pth --epochs 30

    # Modified KD: teacher → MobileNetV2
    python main.py --task distill --model mobilenet --distill_mode modified \\
                   --teacher_path best_teacher.pth --epochs 30

    # Count FLOPs (requires ptflops: pip install ptflops)
    python main.py --task classify --model resnet --count_flops --mode test
