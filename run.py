import argparse
import os
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata

from predictor_engine import DetectronPredictorEngine

def main():
    parser = argparse.ArgumentParser(description="Run Detectron2 inference.")
    parser.add_argument("input_path", type=str, help="Path to the input image file.")
    parser.add_argument("output_path", type=str, help="Path where the output image will be saved.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input image '{args.input_path}' does not exist.")
        return

    try:
        engine = DetectronPredictorEngine()
    except Exception as e:
        print(f"Error: {e}")
        return
      
    image = cv2.imread(args.input_path)
    if image is None:
        print(f"Error: Could not parse or open '{args.input_path}'.")
        return

    outputs = engine(image)
    instances = outputs["instances"].to("cpu")

    if len(instances) > 0:
        print(f"Detected {len(instances)} instance(s). Rendering mask overlays...")
        
        custom_metadata = Metadata()
        custom_metadata.set(thing_classes=["Polyp"])

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(rgb_image, metadata=custom_metadata, scale=1.2)
        out = visualizer.draw_instance_predictions(instances)

        output_image = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
    else:
        print("No targets found by the model. Exporting original image blank.")
        output_image = image

    cv2.imwrite(args.output_path, output_image)
    print(f"Results successfully saved to: {args.output_path}")

if __name__ == "__main__":
    main()
