# neurotech-objclassifier

To build a mind-controlled quidditch-playing drone, we must first realize that it's _very_ (difficult-difficult-lemon-difficult) hard to determine exactly where a player might want to throw a ball. We can assume that players want to pass balls between drones and that first requires us to track and predict balls

Classification and tracking has ~95% accuracy. The video below is a nice demo of the real-time classification of the ball.

![An awesome gif of the machine classifying a ping pong ball](/drone.gif)

Prediction is done with quadratic regression. I'm currently making an EKF (extended kalman filter) to track the ball. I could just implement a g-h filter, but I think I might just go for all the marbles with the kalman filter. (I also want to get some practice building filters after reading @rlabbe's filters book) :)

We calculate the 3D position of the projectile with camera projection:

![An epic graph of the machine plotting the world coordinates of a ping pong ball](/first_correct_coordinate_graph.png)

### What's next for the drone:

- [ ] Build a simulation framework to test the entire system OR find an online drone simulator I can use
- [ ] Implement the Kalman Filter to predict the ball's position (and velocity)

An First-Person view stream from the drone would be very useful for controlling the drone, and I built the streaming service in Unity [here](http://www.github.com/dhrumilp15/UnityVRStreaming).
