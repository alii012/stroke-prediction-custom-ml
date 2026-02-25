# My Stroke Prediction Project (Custom ML)

### Check out the live app here: [https://stroke-prediction-custom-ml-b2h3jd4q3qfw6mrwm42xbx.streamlit.app/]

## What is this?
I built this project to see if I could predict stroke risk using a dataset from Kaggle. Instead of just using a library like Scikit-Learn to do all the work, I decided to build the Logistic Regression model from scratch using NumPy. 

I did this because I wanted to really understand how Gradient Descent works behind the scenes, especially when dealing with "imbalanced data" (where most people in the data didn't have a stroke).

## How I built it
* **The Math:** I wrote my own Gradient Descent loop. Since strokes are rare in the data, I added a "weight" to the stroke cases. I told the model that missing a stroke is 8 times worse than a false alarm.
* **The Features:** I realized age and glucose levels are huge factors, so I created new features like `age squared` to help the model see patterns better.
* **The App:** I used Streamlit to turn my saved model into a website where anyone can move sliders to see their "probability" score.

## My Results
After 10,000 iterations of training, the model hit about 82% Recall. This means it's pretty good at catching the high-risk cases, which is what matters most in health.

## Tech I used:
Python, NumPy, Pandas, and Streamlit for the deployment.
