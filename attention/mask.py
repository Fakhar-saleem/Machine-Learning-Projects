import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_idx = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_idx is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Run the model
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    outputs = model(**inputs, output_attentions=True)

    # Top-K mask predictions
    mask_logits = outputs.logits[0, mask_idx]
    top_tokens = tf.math.top_k(mask_logits, K).indices.numpy()
    for tok in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([tok])))

    # Convert input IDs back to token strings
    input_ids = inputs["input_ids"][0].numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Generate attention diagrams
    visualize_attentions(tokens, outputs.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the single mask token in inputs["input_ids"][0],
    or None if not found.
    """
    ids = inputs["input_ids"][0].numpy().tolist()
    try:
        return ids.index(mask_token_id)
    except ValueError:
        return None


def get_color_for_attention_score(attn_score):
    """
    Map attn_score in [0,1] to a gray RGB tuple (v,v,v) where
    v=0→black, v=255→white.
    """
    # ensure Python float
    s = float(tf.clip_by_value(attn_score, 0.0, 1.0).numpy())
    v = int(round(s * 255))
    return (v, v, v)


def visualize_attentions(tokens, attentions):
    """
    For each layer and each head in attentions (a tuple of Tensors),
    call generate_diagram with 1-based layer and head indices.
    """
    # attentions: tuple length=L, each shape (1, H, T, T)
    for layer_idx, layer_attn in enumerate(attentions):
        num_heads = layer_attn.shape[1]
        for head_idx in range(num_heads):
            # take the (T×T) matrix for batch=0, head=head_idx
            attn_matrix = layer_attn[0, head_idx].numpy()
            generate_diagram(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                attn_matrix
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Draw and save a grid of attention_weights (T×T) for one head,
    labeling rows/cols by tokens.
    """
    size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (size, size), "black")
    draw = ImageDraw.Draw(img)

    # Draw token labels (columns rotated, rows normal)
    for i, tok in enumerate(tokens):
        # Column label (rotated 90°)
        tmp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(tmp)
        d2.text(
            (size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            tok, fill="white", font=FONT
        )
        img.paste(tmp.rotate(90), mask=tmp)
        # Row label
        _, _, w, _ = draw.textbbox((0, 0), tok, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - w, PIXELS_PER_WORD + i * GRID_SIZE),
            tok, fill="white", font=FONT
        )

    # Draw attention cells
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            y = PIXELS_PER_WORD + i * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save as PNG
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
