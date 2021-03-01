# neurotech-objclassifier

To build a mind-controlled quidditch-playing drone, we must first realize that it's _very_ (difficult-difficult-lemon-difficult) hard to determine exactly where a player might want to throw a ball. We can assume that players want to pass balls between drones and that first requires us to track and predict balls.

I stream developing my projects on [Twitch/dhrumilp15](https://twitch.tv/dhrumilp15)! Come visit sometime!

![An awesome gif of the machine classifying and predicting the path of a ping pong ball](/predicted_path.gif)

Prediction is done with quadratic regression and RANSAC. I'm currently making an EKF (extended kalman filter) to track the ball. I could just implement a g-h filter, but I think I might just go for all the marbles with the kalman filter. I've also been looking at Kunyue Su and Shaojie Shen's "Catching a Flying Ball with a Vision-Based Quadrotor".

We calculate the 3D position of the projectile with camera projection and predict its path using RANSAC and a quadratic regression model:

![An epic plot of the world coordinates and predicted path of a ping pong ball](/predicted_ball_path.png)

Classification and tracking has ~95% accuracy. The video below is a nice demo of the real-time classification of the ball.

![An awesome gif of the machine classifying a ping pong ball](/drone.gif)

### What's next for the drone:

- [ ] Build a simulation framework to test the entire system OR find an online drone simulator I can use
- [ ] Implement the system in an IRL drone

An First-Person view stream from the drone would be very useful for controlling the drone, so I built a VR FPV streaming service in Unity [here](http://www.github.com/dhrumilp15/UnityVRStreaming).
