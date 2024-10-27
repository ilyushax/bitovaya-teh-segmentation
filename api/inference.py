import cv2
from uuid import uuid1
import albumentations as A
import numpy as np
import onnxruntime
from copy import deepcopy

ort_session = onnxruntime.InferenceSession(
    "models/segm_model.onnx", providers=["CPUExecutionProvider"]
)

TRANSFORMS = A.Compose([
    A.Resize(384, 384),
    A.Normalize(),
])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


async def predict(file):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_pred = apply_model(deepcopy(img))
    masked_img = mask_image(img, mask_pred)

    output_file_path = f"{uuid1}_masked_image.png"
    cv2.imwrite(output_file_path, masked_img)

    return output_file_path


def apply_colored_mask(image, mask, color):
    color_mask = np.expand_dims(mask, 2)
    color_mask = np.repeat(color_mask, 3, 2)
    color_mask *= np.array(color).astype(np.uint8).reshape(1, 1, 3)
    return cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)


def apply_model(img):
    transformed_data = TRANSFORMS(image=img)
    transformed_img = transformed_data['image']

    model_input = np.transpose(transformed_img, (2, 0, 1))
    model_input = np.expand_dims(model_input, 0)

    pred = ort_session.run(
        None, {ort_session.get_inputs()[0].name: model_input}
    )
    pred = np.squeeze(pred)
    pred = sigmoid(pred)

    return pred


def mask_image(img, mask_pred):

    masks = [mask_pred[i, :, :] for i in range(4)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for mask, color in zip(masks, colors):
        mask_resized = cv2.resize(
            mask, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_binary = np.where(mask_resized > 0.5, 1, 0).astype(np.uint8)
        img = apply_colored_mask(img, mask_binary, color)

    return img
