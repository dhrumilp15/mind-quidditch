# neurotech-objclassifier

To build a mind-controlled quidditch-playing drone, we must first understand that it's _a little_ (think difficult difficult lemon difficult) hard for our machines to accurately determine exactly where we want to throw balls from an EEG. That's why this repo exists: We help our drone catch and launch balls using computer vision.

Classification and tracking has ~95% accuracy. The video below may or may not be a testament to that, but I'm mainly posting it so that you can see what it looks like.

Prediction is done with quadratic regression. I'm currently working on making an EKF (extended kalman filter) to track the ball, using the quadratic regression as the predicting function. I could quickly implement a g-h filter, but I think I might just go for all the marbles with the kalman filter.

### What's next for the drone:

- [ ] Build a simulation framework in which I can test the entire drone's system OR find an online drone simulator I can use
- [ ] Implement the Kalman Filter to predict where the ball's position (and velocity)
- [ ] Either build a separate repo or organization? that handles telling where the drone to go from a BCI (Brain Computer Interface)

An First-Person view stream from the drone would be very useful for controlling the drone, and I built the streaming service in Unity [here](http://www.github.com/dhrumilp15/UnityVRStreaming).
