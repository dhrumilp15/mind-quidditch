# neurotech-objclassifier

This is a computer vision pipeline to track and predict projectile trajectory! Prediction uses RANSAC and Quadratic Regression and classification uses a colour mask and a circle hough transform.

![An awesome gif of the machine classifying and predicting the path of a ping pong ball](/docs/predicted_path.gif)

I stream developing my projects on [Twitch/dhrumilp15](https://twitch.tv/dhrumilp15)! Come visit sometime!

This is also the computer vision stack for a mind-controlled quidditch-playing drone! Although the drones are controlled by players, we realize that it's *very* (difficult-difficult-lemon-difficult) hard to determine exactly where a player might want to throw a ball. We *can* however, make reasonable assumptions about players' behaviour! That allows us to keep players in control and let computers handle minute adjustments to ensure a smooth experience. Teaching the computer to handle minute adjustments requires tracking and predicting projectile trajectory.

## How do I use this?

Install the necessary libraries: `pip install -r requirements.txt`

### Classifier
To run the ball classifier: `python BallClassifier.py`

The ball classifier accepts a few arguments:
- `-v` or `--video`: If you have a video source that you'd like to run through the ball classifier, provide the id or file location here
- `-d` or `--debug`: This flag tells the system whether to display a stream of the system in action

### Predictor

To run the trajectory predictor: `python TrajectoryPredictor.py`

The trajectory predictor accepts a few arguments:
- `-v` or `--video`: If you have a video source that you'd like to run through the trajectory predictor, provide the id or file location here
- `-d` or `--debug`: This flag tells the system whether to display a stream of the system in action and plots for each x,y,z component of the estimated global positions and trajectory.

## How does this work?

### Prediction

We calculate the 3D position of the projectile with camera projection and predict its path using **RANSAC** and **Quadratic Regression**:

![An epic plot of the world coordinates and predicted path of a ping pong ball](/docs/predicted_ball_path.png)

I'm thinking about making an EKF (extended kalman filter) to track the ball. I could just implement a g-h filter, but I think I might just go for all the marbles with the kalman filter. I've also been looking at *Kunyue Su and Shaojie Shen's "Catching a Flying Ball with a Vision-Based Quadrotor" (2017)*.

### Classification

Classification has ~95% accuracy. Classification uses a **colour mask** for a center and radius and a **circle hough transform** to refine the center and radius.

![An awesome video of the machine classifying a ping pong ball](/docs/drone.gif)

## What's Next?

- [ ] Build a simulation framework to test the entire system OR find an online drone simulator I can use
- [ ] Implement the system in an IRL drone

An First-Person view stream from the drone would be very useful for controlling the drone, so I built a [VR FPV streaming service in Unity](http://www.github.com/dhrumilp15/UnityVRStreaming).
