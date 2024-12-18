import tkinter as tk
from customtkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from openai import OpenAI  # OpenAI Python library to make API calls
import requests  # used to download images
from io import BytesIO
from IPython.display import display
import keywords as kw 
from rembg import remove

client = OpenAI(api_key="KEY")

# Set the appearance mode (light or dark) and color theme
set_appearance_mode("light")  # You can also use "dark" or "system"
set_default_color_theme("blue")  # Options: "blue", "dark-blue", "green", etc.

# Create the main window
root = CTk()  # Use CTk instead of tk.Tk for a custom window
root.title("Outfit Input Form")

# Set the size of the window
root.geometry("800x600")

# Create frames for each "tab"
tab1 = CTkFrame(root)
tab2 = CTkFrame(root)


selected_top = None
selected_bottom = None


def generateImages(keywords):
  # set the prompt
  prompt = f"""Generate hyperrealistic images based on the provided keywords. Here are the key words: {keywords}. 
  Make sure the item aligns with the specified theme or aesthetic, capturing realistic fabric, 
  texture, and proportions. The image should ONLY contain a SINGLE piece of clothing. The image should ensure the focus remains on the item itself the item should NOT BE DISPLAYED ON A MODEL. 
  Avoid background distractions and emphasize subtle variations in fabric and design elements to 
  achieve hyperrealism."""

  print(prompt)

  # call the OpenAI API
  generation_response = client.images.generate(
      model = "dall-e-2",
      prompt=prompt,
      n=3,
      size="1024x1024",
      response_format="url",
  )

  urls = []
  for i in range(3):
      generated_image_url = generation_response.data[i].url  # extract image URL from response
      urls.append(generated_image_url)
  return urls
    

# Function to show tab1
def show_tab1():
    tab1.pack(fill='both', expand=True)
    tab2.pack_forget()  # Hide tab2

# Function to show tab2
def show_tab2():
    tab2.pack(fill='both', expand=True)
    tab1.pack_forget()  # Hide tab1

# Initially show tab1
show_tab1()

file_path = ""  # Initialize file_path globally

# Define a function to handle file upload
file_uploaded = False  # This will track if a file is uploaded
def upload_file():
    global file_uploaded, file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.configure(text="File selected: " + file_path)
        file_uploaded = True  # Mark that a file has been uploaded
    else:
        file_label.configure(text="No file selected")
        file_uploaded = False  # No file uploaded

top_keywords = ""
bottom_keywords = ""

top_urls = []
bottom_urls = []

## Function to handle submission
def submit():
    global top_keywords, bottom_keywords, top_urls, bottom_urls
    user_input = entry.get()
    # Check if either text input or file upload is provided
    if user_input or file_uploaded:
        # Call the function from keywords.py with both inputs
        if user_input and file_uploaded:
            top_keywords, bottom_keywords = kw.keywordGen(user_input, file_path)
        elif user_input:
            top_keywords, bottom_keywords = kw.keywordGen(user_input, "")
        elif file_uploaded:
            top_keywords, bottom_keywords = kw.keywordGen("", file_path)
        top_urls = generateImages(top_keywords)
        #bottom_urls = generateImages(bottom_keywords)
        displayURL(top_urls)
        # Move to the next tab (the blank tab) if input or file is provided
        show_tab2()  # Show tab2
    else:
        messagebox.showwarning("No Input", "Please enter an outfit or upload a file!")

# LABEL AT THE TOP OF PAGE
prompt_label = CTkLabel(tab1, text="Enter an outfit", font=("Helvetica", 16, "bold"))  # Use CTkLabel for custom label
prompt_label.pack(pady=20)

# TEXT BOX
entry = CTkEntry(tab1, width=250, font=("Helvetica", 12))  # Use CTkEntry for a custom entry box
entry.pack(pady=10)

# FILE UPLOAD BUTTON AND SHOWS PATH 
upload_button = CTkButton(tab1, text="Upload a File", command=upload_file, font=("Helvetica", 12))  # Use CTkButton for custom button
upload_button.pack(pady=10)

file_label = CTkLabel(tab1, text="No file selected", font=("Helvetica", 10))  # Use CTkLabel for file label
file_label.pack(pady=5)

# SUBMIT BUTTON (Moved below the upload file button)
submit_button = CTkButton(tab1, text="Submit", command=submit, font=("Helvetica", 12))  # Use CTkButton for custom button
submit_button.pack(pady=10)

# Tab 2
hello_label = CTkLabel(tab2, text="Welcome to the virtual fitting room!!", font=("Helvetica", 16, "bold"))  # Add a message for tab 2
hello_label.pack(pady=10)

# TOPS SECTION
tops_label = CTkLabel(tab2, text="TOPS", font=("Helvetica", 16, "bold"))
tops_label.pack(pady=(10, 5))  # Add some padding

# Create a frame for tops
tops_frame = CTkFrame(tab2)
tops_frame.pack()

# Load images for tops
tops_images = []
tops_buttons = []
selected_top = None
selected_top_image = None

def select_top(index):
    global selected_top, selected_top_image 
    if selected_top is not None:
        tops_buttons[selected_top].configure(fg_color="white")  # Reset previous selection
    if selected_top == index:
        selected_top = None  # Deselect if clicked again
        selected_top_image = None
    else:
        selected_top = index  # Select the new top
        selected_top_image = tops_images[index]
    if selected_top is not None:
        tops_buttons[selected_top].configure(fg_color="lightgreen")  # Highlight selected



# BOTTOMS SECTION
#bottoms_label = CTkLabel(tab2, text="BOTTOMS", font=("Helvetica", 16, "bold"))
#bottoms_label.pack(pady=(20, 5))  # Add some padding

# Create a frame for bottoms
#bottoms_frame = CTkFrame(tab2)
#bottoms_frame.pack()

# Load images for bottoms
#bottoms_images = []
#bottoms_buttons = []
#selected_bottom = None

#def select_bottom(index):
    #global selected_bottom
    #if selected_bottom is not None:
        #bottoms_buttons[selected_bottom].configure(fg_color="white")  # Reset previous selection
    #if selected_bottom == index:
        #selected_bottom = None  # Deselect if clicked again
    #else:
        #selected_bottom = index  # Select the new bottom
    #if selected_bottom is not None:
        #bottoms_buttons[selected_bottom].configure(fg_color="lightgreen")  # Highlight selected

def show_tab3():
    tab3.pack(fill='both', expand=True)
    tab1.pack_forget()
    tab2.pack_forget()



def displayURL(top_urls):
    for i, url in enumerate(top_urls):
        response = requests.get(url)  # Get the image directly from the URL
        img = Image.open(BytesIO(response.content)).resize((100, 100), Image.LANCZOS)  # Open image in memory using BytesIO
        img = remove(img)
        tops_images.append(ImageTk.PhotoImage(img))
        button = CTkButton(tops_frame, image=tops_images[i], command=lambda idx=i: select_top(idx), text="", width=120)
        button.pack(side=tk.LEFT, padx=10)
        tops_buttons.append(button)

    # Replace these image paths with your actual image paths
    #for i, url in enumerate(bottom_urls):
        #response = requests.get(url)  # Get the image directly from the URL
        #img = Image.open(BytesIO(response.content)).resize((100, 100), Image.LANCZOS)  # Open image in memory using BytesIO
        #bottoms_images.append(ImageTk.PhotoImage(img))
        #button = CTkButton(bottoms_frame, image=bottoms_images[i], command=lambda idx=i: select_bottom(idx), text="", width=120)
        #button.pack(side=tk.LEFT, padx=10)
        #bottoms_buttons.append(button)

def try_on():
    if selected_top is None:
        messagebox.showwarning("Selection Required", "Please select a top before proceeding.")
    else:
        show_tab3()  # Move to tab3 when both selections are made


# "Try On" Button
try_on_button = CTkButton(tab2, text="Try On!", command=try_on, font=("Helvetica", 12))
try_on_button.pack(pady=(20, 10))

tab3 = CTkFrame(root)

def show_tab3():
    tab3.pack(fill='both', expand=True)
    tab1.pack_forget()
    tab2.pack_forget()

    if selected_top_image is not None:
        final_image_label = CTkLabel(tab3, image=selected_top_image, text="", width=200)  # Display selected image
        final_image_label.pack(pady=20)

final_label = CTkLabel(tab3, text="Here's your outfit!", font=("Helvetica", 16, "bold"))
final_label.pack(pady=20)

# Run the application
root.mainloop()
