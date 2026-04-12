Before using the library, make sure that you downloaded the CIFAR-10-C data in this directory as data

This library were created to observe robustness of previous homewrok model and new model trained with AugMix.

Test accuracy of previous and new model were compared.
GradCam and t-SNE analysis of these two models were performed.
Augmix architecture also were tested in transfer learning part in the previous homework.


Usage examples::

    # Robustness evaluation (clean + CIFAR-10-C)
    python main.py --task robustness --model_path best_resnet.pth

    # Train AugMix model then evaluate on clean + CIFAR-10-C
    python main.py --task augmix --mode both --save_path augmix_resnet.pth --save_plots

    # Only test an existing AugMix checkpoint
    python main.py --task augmix --mode test --save_path augmix_resnet.pth

    # PGD evaluation on a standard checkpoint
    python main.py --task pgd --model_path best_resnet.pth --pgd_n_samples 500

    # Grad-CAM visualization
    python main.py --task gradcam --model_path best_resnet.pth

    # t-SNE of adversarial features
    python main.py --task tsne_adv --model_path best_resnet.pth --tsne_n_samples 300

    # Distillation: AugMix teacher → SimpleCNN student
    python main.py --task distill_augmix \\
                   --teacher_path augmix_resnet.pth \\
                   --student_model cnn --distill_mode standard \\
                   --save_path best_student_cnn.pth

    # Adversarial transferability
    python main.py --task transfer_pgd \\
                   --teacher_path augmix_resnet.pth \\
                   --student_path best_student_cnn.pth \\
                   --student_model cnn

