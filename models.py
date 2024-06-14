from transformers import AutoImageProcessor, AutoModelForObjectDetection


def get_huggingface_model(config: dict):
    processor = AutoImageProcessor.from_pretrained(config["model_parameters"]["model_name"])
    model = AutoModelForObjectDetection.from_pretrained(config["model_parameters"]["model_name"])

    return model, processor
