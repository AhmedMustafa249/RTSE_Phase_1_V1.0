# Assignment 2

# SPEEDTRIALS2D: Real-Time and Concurrent Software Development

Assessment Mark: 20% 

<table><tr><td>Course</td><td>SECJ 4423 Real-Time Software Engineering</td><td>Assignment</td><td>Assignment 2</td></tr><tr><td>Topic</td><td>Concurrent Programming, RTOS Threading, and Real-Time Schedulability Analysis</td><td>Reference</td><td>Selected Autonomous Vehicle Case-Study Research Paper (2022–2026)</td></tr><tr><td>Outputs</td><td>Submission: Code, GitHub &amp; Individual Video, Technical Paper, Group Presentation VideoOthers: Game Participation, Discord Group Learning and Game Score</td><td>Submission</td><td>Turnitin (for Academic Integrity Verification) &amp; E-learning Portal</td></tr></table>

## 1. Assignment Brief

This assignment is to develop and analyze a concurrent, multi-threaded control system for a self-driving car simulation using RTOS principles to investigate how task scheduling, priority, and timing requirements directly affect the vehicle's real-time responsiveness and stability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-17/d36c8bdc-ee72-4365-b51a-6eb43c58fa6d/e8497f3ef297dfd4077b81bddd11a8483aa466d7573677a0283dee10116921c6.jpg)


## 2. Learning Outcomes

I. Apply appropriate real-time software engineering methods and concurrent programming tools to develop a responsive self-driving control system within a simulated environment, utilizing multi-threaded programming. (CLO2) 

II. Predict the timing performance of a real-time software design using schedulability analysis and benchmarking, while incorporating comparative research and system redesign proposals based on existing self-driving car literature. (CLO3) 

III. Demonstrate the ability to adapt effectively in a team to apply real-time software engineering knowledge in developing a medium-scale real-time application, including the production of a technical paper and group demonstration. (CLO4) 

## 3. Task Requirements

## 3.1 SPEEDTRIALS2D Programming

To develop a multi-threaded control system that mimics the uC/OS-II RTOS structure with the following requirements: 

• Vehicle Control Programming: Write Python code to manage movement and environmental interaction. Implement logic for lane switching, token reaction, and speed optimization while maintaining stability. 

• Concurrent Design & Programming: Apply concurrency to handle real-time image processing, decision-making, and command execution simultaneously under strict timing constraints for peak performance. 

• Game Environment Understanding: Master track mechanics and dynamic events. Build robust systems that adapt to obstacles, token effects, and varying environmental conditions. 

## 3.2 Game Challenge

All groups must participate in an online game session to evaluate their control systems under live, competitive conditions. Teams must fully master the game environment. This live session serves as the ultimate real-time test of your system's stability, latency, and decision-making under strict constraints. 

## 3.3 Technical Support & Online Mentorship

To ensure technical success during the development of SpeedTrials2D, a dedicated support system has been established on the class Discord platform. 

Postgraduate Technical Assistants for Online Consultation: Three postgraduate students will be available to provide technical guidance on RTOS implementation, concurrency, and simulation troubleshooting. Students are encouraged to use the Discord channels to discuss complex issues with mentors. 

Peer-to-Peer Learning: While the assistants provide expert oversight, the Discord server serves as a real-time collaborative space for teams to share insights on overcoming common simulation bottlenecks. This learning process is measured through your team’s active participation in brainstorming and collective problem-solving on the official Discord platform. 

## 3.4 A Self-Driving Car Case Study

Drawing from your team’s development experience with the SpeedTrials2D self-driving simulation, you are required to transition from a virtual environment to a physical real-time design proposal. Write a report based on the following task. 

## a. Research & Case Study Selection

Identify and review a research paper published between 2022 and 2026 that features a real-world selfdriving or autonomous vehicle case study. The selected paper must provide detailed insights into: 

● Software Architecture: Specifically, the Perception-Decision-Actuation pipeline. 

● Real-Time Scheduling: The specific approach used to manage tasks using such as Rate Monotonic Scheduling (RMS), Earliest Deadline First (EDF) or others. 

## b. Concurrent Design Improvement Proposal

Based on your team’s experience with uC/OS-II or multi-threaded Python development, propose a system redesign that addresses the following: 

● High-Speed Optimization: How would you modify your concurrent design to maintain stability and safety in higher-speed environments where reaction windows are significantly smaller? 

Sensor Stack Scaling: Based on the rigorous requirements of your chosen case-study paper, describe how your architecture or design would adapt to more complex sensor data (adapted from the case study e.g., integrating LiDAR or Radar) while managing the increased computational load and interrupt frequency. 

## c. Real-World Schedulability Testing

In a professional vehicle development environment, "manual observation" is insufficient. Detail how your Real-Time Scheduling Analysis would be rigorously tested on a physical car, focusing on: 

● Worst-Case Execution Time (WCET): How tasks would be benchmarked to ensure they never miss a deadline. 

● Hardware-in-the-Loop (HiL): The process of testing your software on actual embedded controllers to measure interrupt latency and resource contention. 

● Safety Thresholds: Identifying the boundary between stable system performance and degraded, unsafe behavior. 

## 4. Task and Marks Distribution – Total 20%

a. SPEEDTRIALS2D Programming (CLO2) 5% 

i. Code (3%) 

ii. GitHub & Individual Technical Contribution Video (2%) 

b. A Self-Driving Car Case Study (CLO3) 10% 

i. Technical Paper - IEEE Style Format (6%) 

ii. Group Presentation video (4%) 

c. Game Participation (CLO4) 5% 

i. Game Attendance (1%) 

ii. Discord Discussion Participation (2%) 

iii. Game Score (2%) 

## 4. Suggested Report Structure

## 1. Introduction

● Overview: Provide an introduction to Real-Time Systems Engineering (RTSE) and the objectives of the self-driving simulation project. 

● Problem Statement: Briefly describe the challenges of maintaining vehicle stability under strict timing constraints in a dynamic environment. 

## 2. Background Study

● Concurrent Programming: Define concurrency and explain its necessity in real-time systems to handle simultaneous tasks like perception and actuation. 

● SpeedTrials2D Simulation: Introduce the Unity-based simulation environment, the vehicle control parameters, and the Python-based embedded system model. 

## 3. RTOS Concurrent Design & Implementation

● Concurrent Requirements: Define the "Perceive-Compute-Actuate" pipeline and identify the specific tasks required for autonomous navigation. 

● Task Architecture (uC/OS-II Modeling): Detail your multi-threaded design, including task priority assignments, periods (ms), and inter-task communication (e.g., Queues/Mutexes). 

## 4. Case Study & Comparative Analysis

● Case Summary: Summarize a 2022–2026 paper on real-world autonomous driving software and its Perception-Decision-Actuation pipeline. 

● Scheduling Review: Identify the paper’s task scheduling approach (e.g., RMS or EDF). 

● Benchmarking: Compare the paper’s architecture and timing constraints against your SPEEDTRIALS2D simulation findings. 

## 5. System Redesign & Real-World Testing

Optimization Proposal: Detail your plan to upgrade the design for high-speed safety and complex sensors (LiDAR/Radar), using Discord assistant feedback. 

Schedulability Validation: Explain how to test this design on a physical car using: 

o WCET: Benchmarking tasks to guarantee deadlines are met. 

o HiL Testing: Running software on actual embedded hardware to measure latency and resource locking. 

o Safety Thresholds: Finding the exact point where system performance becomes unsafe. 

## 6. Conclusion

● Summarize key insights gained regarding the relationship between task scheduling and real-time responsiveness. 

● Reflect on how simulation-based testing prepares for physical real-time application development. 

## 7. References

List all citations in IEEE or APA academic format. 

● Include the Mandatory AI Disclosure detailing how AI tools (if any) and Discord postgraduate assistants aided your development. 

## Technical Mandates

IEEE paper format 

Plagiarism: Final submission must show < 20% similarity on Turnitin. 

● AI Content: AI-generated content must be < 20%. 

Submission: Official upload via Turnitin; no resubmissions permitted. 

● AI Usage Declaration: Students must include a brief statement in their technical paper identifying any AI tools used during this assignment. Clearly state the tool's name and how it aided your programming or writing process. 


5. Required Deliverables


<table><tr><td>Deliverable</td><td>Marks</td><td>Minimum Content</td><td>Suggested Format</td></tr><tr><td>SPEEDTRIALS2D Code</td><td>3%</td><td>Complete Python script mimicking uC/OS-II task structure (Perceive-Compute-Actuate) with lane-switching, token reaction, and thread safety.</td><td>Python files (.py)</td></tr><tr><td>GitHub &amp; Individual Video</td><td>2%</td><td>Short video showing personal code contribution, logic explanation, live functional proof, and GitHub commit history and prove how it functions within your team&#x27;s final codebase.</td><td>Video File / Link (MP4/Stream)</td></tr><tr><td>Technical Paper</td><td>6%</td><td>Report covering: Project intro/concurrency background, uC/OS-II modeling (priorities/periods), 2022–2026 case study benchmarking, and a redesign proposal (LiDAR/Radar, WCET, HiL testing).</td><td>IEEE Format PDF via Turnitin (&lt;20% Turnitin &amp; AI content)</td></tr><tr><td>Group Presentation Video</td><td>4%</td><td>Group walkthrough explaining project findings, simulation implementation, and the physical real-time design proposal for the case study.</td><td>Video File / Link</td></tr><tr><td>Game Participation</td><td>1%</td><td>Active group presence and control system execution during the live online simulation match.</td><td>Session credit</td></tr><tr><td>Discord Group Learning</td><td>2%</td><td>Evidence of group brainstorming, problem-solving, and technical consultation with mentors/postgraduates on Discord.</td><td>Verified Discord engagement logs</td></tr><tr><td>Game Score</td><td>2%</td><td>Performance mark based on the maximum distance achieved by the vehicle during live tracking.</td><td>Simulator leaderboard output</td></tr></table>

## 6. Submission Guidelines

• Submit all deliverables through the e-learning platform within the deadline. 

• Use academic writing and cite all external sources properly. 

• Screenshots, diagrams, and visual illustrations must be clear and readable. 

• Forum participation is individual even though the main assignment is completed in groups. 

• Academic honesty is compulsory. Plagiarism, unreferenced copying, or recycled work is not acceptable. 

## 7. Important Date

SPEEDTRIALS2D Technical Workshop Session 29 May 2026 

Individual Video 17 June 2026 

Game Session 20 June 2026 

Group Video 24 June 2026 

Final Report 24 June 2026 