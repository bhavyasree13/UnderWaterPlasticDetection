from ultralytics import YOLO

def main():
    model = YOLO('yolov10n.pt')

    model.train(
        data='dataset/data.yaml',
        epochs=120,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
