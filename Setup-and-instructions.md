## RTSE 2026 LAB

## SPEEDTRIALS20

Real-Time and Concurrent Software Development Faculty of Computing,UTM, Skudai 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/9a94f1cf07a7abf6b78260053c9b368e2b4dbe9f4b83c4faa7c6dea77a8d301e.jpg)


## Join the Discord

Make sure to rename your discord to : 

## TeamName_Section_Ful lName

After you rename an admin will assign a role and a team channel. 

Teams must display at least once a week collaboration on the RTSE Lab. 

## 1.Download GitHub Release

1.Goto:https://github.com/MokhtarOuardi/Hack2Drive_2026/releasesand download the latest Phase1 

2.Download the latest releasearchive (ZiPor tar.gz). 

3.Extract the contents to your working directory. 

Youshould see a structure similarto: 

```txt
RTSE_Phase_1_VN.N/
| — SpeedTrials2D/
| — test_communication.py
| — sample_drive.py
| — requirements.txt 
```

## 2.Install Python Requirements

Ensureyou have Python 3.8+installed. 

Opena terminal inside the project folder and run: 

pipinstall-rrequirements.txt 

## 3.Run the Simulator

Launch the environment: 

./SpeedTrials2D/SpeedTrials2D.exe 

OnWindows,you can also double-click the executable. 

Makesure the simulation windowopens correctly beforeproceeding. 

## Verification

After completing the previous setup steps, the simulator window should appear as shown. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/7527b3f1f4496ca9721ef6c3c864a046984af25a864053ae64cd5d2e284025e0.jpg)


Simulator Ready 

Pressing R resets the simulation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/970d77b241bbca01ee696efdd2465468c4dac1964137509fead17d111c1f7b2f.jpg)


## Attention !

Read the comments in sample_drive.py. It is important to follow the RTOS format in order for your code to be eligible. You must follow scheduling instructions. 

## 4.Understand Example Scripts

Beforecoding yourownsolution,review theprovided scripts: 

## test_communication.py

·Demonstrates how to send and receivedata from the simulator. 

·Use itto test if you canreceive thecamera feed from the simulation 

·Yourcode shouldn't followthiscode format 

## sample_drive.py

Basic example of controlling the car. 

·This isthe code format youmust usefor RTOS compliance 

·Shows howto read,process and control 

Taketimetounderstandhowinputismappedtovehiclemovement. 

## Verification

Running either python scripts along with the simulation should display the following windows. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/1fa07c14b2acf34bf8f5e373c0500b0737e0d4a75ff4da2253e65fc7afa5c275.jpg)


## Script Ready

As long as you follow the RTOS code structure and comply to the perceive, compute and actuate pipeline there is no limits to the algorithm you implement. Game abuses or Cheats aren’t allowed, your code will be reviewed. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/2231d2eef8a204f9a761a2a6808f7113b2626f717bf3a1709dcb540bbc42512f.jpg)


## Getting started with computer vision :

Link : 

https://learnopencv.com/getting -started-with-opencv/ 

## 5. Detect Token Using OpenCV

Theseareexamples,youareallowedandencouragedtouseanylibraries that suits you.Youmay traina YOLO modelforexample. 

Exampleapproach: 

```python
import cv2
import numpy as np

frame = cv2.imread("frame.png")  # In your code you must replace this with the
    # frame coming from SpeedTrials2D.exe
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Example: detect a red token
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    token_center = (x + w//2, y + h//2)

    cv2.circle(frame, token_center, 5, (0, 255, 0), -1) 
```

## Attention !

When sending steering action to the control port you need to tap the steering input. 

E.g. Moving 1 lane to the right : 

```txt
Steering_input = 0.0
wait()
Steering_input = 1.0
wait()
Steering_input = 0.0 
```

## 6.Send Control Commands

Yousend continuous control signals tothesimulator usingasocket connection. 

You willcontrol the carusing two floating-point values: 

·steering_input:-1.0(left）to1.0(right) 

·acceleration_input:-1.0(reverse) to1.0(forward) 

## Control Function

```python
import struct

def send_controls_task():
    # This is where you send the control commands to the car using the control_
    global control_conn
    if control_conn is None:
    return

    # these are the variables used to control the car
    # steering_input: -1.0 to 1.0 (left to right)
    # acceleration_input: -1.0 to 1.0 (reverse to forward)

    steering_input = 0.0
    acceleration_input = 1.0  # default: always move forward

    try:
    # Pack and send the control command
    data = struct.pack('ff', steering_input, acceleration_input)
    control_conn.sendall(data)
    except Exception as e:
    print(f"Control send error: {e}")
    control_conn = None 
```

This is a simple demo with detect green token logic. Your turn to code the perfect logic that goes the furthest in 60 sec. 

Good Luck & Have Fun =) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/c5fb915da0052a95c0e3c8140644fa3df2de0dbc65db3b540b4e5a7f3ada49f6.jpg)


## Objective

Maximize distance travelled in 60 sec. 

Collect the least amount of red and yellow tokens 

Survive RTSE challenges & random events. 

## Core Gameplay

Python control a Unity simulation using real-time camera input. Performance tied to efficiency, reaction, and stability under corruption. 

## Scoring Factors

Distance covered 

Consistency over 10 runs 

Event success rate 

Penalty minimization 

## Dynamic Events & Tokens

● Green: +10% speed | ● Red: -20% speed 

● Yellow: Randomized : 

20% → Next token type hidden 

20% → Tokens invisible for 5s 

20% → Camera input delay (5s) 

20% → Action output delay (5s) 

20% → Corrupted camera input (5s) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/0e44730e8193fb62e5842bb22bde3eb6ff4ba0ed9b496dce896792aec242f00b.jpg)


## Events

Trailing Car: Must switch lanes before collision or -50% speed. 

● Police: Catch next Red Token or -50% speed. 

Low Brightness: Turn the light ON or All tokens Yellow. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/9e44ff90e71254f57d9b2008f2291468765812b46a7aa79afbb19ca69ea93402.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/2851a07e016ad6e0504d1811d38023f861602f28c4360f470dae81b2b32caa93.jpg)


WIN CONDITION: HIGHEST DISTANCE SCORE AFTER 60s 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/aeb41664b890589064084368c1e8d0ea01ec042b36333c8bd917e9c22b4f3f33.jpg)


## Upload Your Weekly Progress

Submit a link for a short demo every Tuesday to show your progress. 

Your leaderboard rank will be used for seeding during the final competition. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/cd16fef2239fa15b1a134fde895cd77398d5c58506f121aeb00af7b33cb227c1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/4d78e7f2e0eecf4a9431c2ecaf06106a75175e7abbef13251a0c1ea58d686921.jpg)


## Github

Everyteammust havea github repowith their algo.Eachmember must commitat least onceperweek.The commitmustincludea valuablecontribution. Youmaykeepyour github privateuntil the assessment phase. 

## ASSESSMENT:INDIVIDUALCONTRIBUTIONVIDEO

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/14346a07dcfa25de7aeca5831827451d924a2d148c10544774319956a9db3b20.jpg)


## Showcase Your Individual Impact

Demonstrateyour specific technical work. 

Explainyourcontribution and itsvalue. 

Illustrateyour understanding of the code. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/f5a82001fb66c3dae5575e2ffd1aad800995223adf43fdd379d46090f9430a64.jpg)


● New Challenges / New Tokens 

● 48 hours to tune your algo & Strategy 

● Live race against other teams 

## THEFINALRACE:TOURNAMENTSTYLE

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/abe3d9f6cac3ac2ca402f45ba52d4287f13e426e7aff571bdcc613622c5df607.jpg)


## Tournament Style :

Challenge your classmates and demonstrate your algo and strategy performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/53934d45f67fafded7d24903c197d606287018163f4f317e0f3c98ec7355c61c.jpg)


## New Challenges :

48 hours before the final race new tokens & events will be added to challenge your adaptability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/444a9d47-81a4-4df1-ae68-e8b024f3a1ee/4ab3a47e0d1c1f1e740a3a9a9401f921b45e5b1601e9074a78828390c19dfb98.jpg)


## Weekly Leaderboard Rank Advantage :

The highest ranked teams in the leaderboard get to select their opponent for round 1 of the competition. 