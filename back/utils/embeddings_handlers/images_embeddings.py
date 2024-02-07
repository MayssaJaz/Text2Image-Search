from PIL import Image


def get_single_image_embedding(image_path, device, model, processor):
    image = processor(
        text=None,
        images=Image.open(image_path).convert("RGB"),
        return_tensors="pt"
    )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # convert the embeddings to list
    embedding_as_list = embedding.squeeze().cpu().detach().numpy().tolist()
    return embedding_as_list


def get_all_images_embedding(images_paths, device, model, processor):
    images_embeddings = [get_single_image_embedding(
        path, device, model, processor) for path in images_paths]
    return images_embeddings