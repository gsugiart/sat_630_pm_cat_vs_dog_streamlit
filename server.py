import streamlit as st
from fastai.vision.all import *

def label_function(f):
    return f[0].isupper()

# load our pre-trained model
cat_vs_dog_model = load_learner("cat_vs_dog.pkl")



# Define a function to make predictions
def predict(image):
    img = PILImage.create(image)  # Use PILImage.create to open the image
    pred_class, pred_idx, outputs = cat_vs_dog_model.predict(img)
    likelihood_is_cat = outputs[1].item()
    if likelihood_is_cat > 0.9:
        return "Cat"
    elif likelihood_is_cat < 0.1:
        return "Dog"
    else:
        return "Not sure... try another picture!"

st.title("Cat vs. Dog Classifier")
st.text("Built by Gamas Chang")



uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    prediction = predict(uploaded_file)
    st.write(prediction)

    # Make predictions on the uploaded image
    # if st.button("Predict"):
    #     prediction = predict(uploaded_file)
    #     st.write(prediction)

# Add a footer
st.text("Built with Streamlit and Fastai")

