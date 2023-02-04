import os
import json
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from utils import decode_base64_to_image, load_label_mapping, map_class_to_label
from utils import extract_model

# download model file from S3 into /tmp folder
extract_model(os.environ['MODEL_S3_URI'], '/tmp')

model = torch.jit.load("/tmp/model.scripted.pt")
model.eval()
predict_transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
topk = 6

response_headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

categories = load_label_mapping("index_to_name.json")


def inference(image: Image) -> Dict[str, int]:
    img_tensor = predict_transforms(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        preds = F.softmax(logits, dim=-1)
    return preds


def handle_request(event, context):
    print(f"Lambda function ARN: {context.invoked_function_arn}")
    print(f"Lambda function version: {context.function_version}")
    print(f"Lambda Request ID: {context.aws_request_id}")

    print("Got event", event)

    img_b64 = event["body"]

    try:
        image = decode_base64_to_image(img_b64)
        image = image.convert("RGB")

        predictions = inference(image)

        probs, classes = torch.topk(predictions, topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()

        print(f"Lambda time remaining in MS: {context.get_remaining_time_in_millis()}")

        class_to_label = map_class_to_label(probs, categories, classes)

        return {
            "statusCode": 200,
            "headers": response_headers,
            "body": json.dumps(class_to_label),
        }

    except Exception as e:
        print(e)

        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({"message": f"Failed to process image: {e}"}),
        }
