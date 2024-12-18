from openai import OpenAI
import base64
import requests

client = OpenAI(api_key="sk-mCr6ErHGUpA61tDTwFcSxTT3-Dc1G8uDmbdPq5uNgtT3BlbkFJorejbLYYc4Nw_7aNCwXpciLnD4rJPSTZDZzcB_hM4A")


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def keywordGen(description=None, image_path=None):
    # If image path is provided, process the image and create base64 encoding
    if image_path:
        base64_image = encode_image(image_path)

        prompt = (
            "Analyze the outfit in the image and split it into top and bottom. "
            "Provide 15 objective, comma-separated keywords for the top and 15 objective, "
            "comma-separated keywords for the bottom. Focus on concrete elements like type of clothing, "
            "patterns, fabrics, accessories, and construction details. For each section (top and bottom), "
            "include one keyword that categorizes the outfit as either casual/streetwear, smart casual, business casual, "
            "or business professional, and another keyword that describes the overall tone (e.g., structured, loose, fitted). "
            "Additionally, incorporate objective keywords that describe the relationship between the top and bottom in terms of "
            "style coordination or contrast. Avoid subjective interpretations and mentioning colors. Format the output as follows: "
            "Top Clothing Description: (keywords) Bottom Clothing Description: (keywords)"
        )

        words = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        keywords = words.choices[0].message.content

        # If description is also provided, modify the keywords based on it
        if description:
            modified_prompt = (
                keywords + f"\nTake these keywords and modify them based on this input: {description}, "
                "output in the format: Top Clothing Description: (keywords) Bottom Clothing Description: (keywords)"
            )

            modified = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": modified_prompt,
                            }
                        ],
                    }
                ],
            )

            modified_keywords = modified.choices[0].message.content
        else:
            # If no description is provided, just use the image-based keywords
            modified_keywords = keywords

    elif description:
        # If no image is provided, but a description is available
        modified_prompt = (
            f"The input is: {description}. Analyze the outfit in the input and split it into top and bottom. "
            "Provide 15 objective, comma-separated keywords for the top and 15 objective, "
            "comma-separated keywords for the bottom. "
            "Output in the format: Top Clothing Description: (keywords) Bottom Clothing Description: (keywords). if the input is only describing only a top piece of clothing or only a bottom piece of clothing, output the keywords for the category it is describing and don't output the other category. for example if the user only describes black jeans print the keywords in the format Bottom Clothing Description: (keywords) and dont print anything else"
        )

        modified = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": modified_prompt,
                        }
                    ],
                }
            ],
        )

        modified_keywords = modified.choices[0].message.content
    else:
        raise ValueError("Either image_path or description must be provided.")

    

    if "Top Clothing Description:" in modified_keywords and "Bottom Clothing Description:" in modified_keywords:
        # Split the modified_keywords into top and bottom clothing descriptions
        split_keywords = modified_keywords.split("Bottom Clothing Description:")
        top_clothing_description = split_keywords[0].replace("Top Clothing Description:", "").strip()
        bottom_clothing_description = split_keywords[1].strip()
    elif "Top Clothing Description:" in modified_keywords:
        # If only Top Clothing Description is present
        top_clothing_description = modified_keywords.replace("Top Clothing Description:", "").strip()
        bottom_clothing_description = ""
    elif "Bottom Clothing Description:" in modified_keywords:
        # If only Bottom Clothing Description is present
        top_clothing_description = ""
        bottom_clothing_description = modified_keywords.replace("Bottom Clothing Description:", "").strip()
    else:
        raise ValueError("Neither Top nor Bottom Clothing Description found in the input.")


    if top_clothing_description.count(',') == 0:
        top_clothing_description = "white t-shirt, short sleeves, crew neck, cotton fabric, relaxed fit, plain design, no patterns, casual style, lightweight, soft texture, pullover style, breathable, minimal details, casual streetwear, loose fit"
    if bottom_clothing_description.count(',') == 0:
        bottom_clothing_description = "black joggers, elastic waistband, drawstring closure, tapered leg, soft fabric, cotton blend, no patterns, casual style, comfortable fit, side pockets, ribbed cuffs, relaxed fit, casual streetwear, lightweight, stretchy material"
    
    return top_clothing_description, bottom_clothing_description
