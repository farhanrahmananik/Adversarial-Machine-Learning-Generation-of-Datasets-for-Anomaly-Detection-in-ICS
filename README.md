# 🧠 Study Project: Adversarial-Machine-Learning-Generation-of-Datasets-for-Anomaly-Detection-in-ICS 
### Part I – Statistical Analysis and Generation of Network Data  
**Chair of IT Security, Winter Semester 2025/2026**  
**Brandenburg University of Technology (BTU Cottbus-Senftenberg)**  
Supervisors: *Asya Mitseva, M.Sc.* and *Prof. Dr.-Ing. Andriy Panchenko*

---

## 📘 Overview
This repository contains the implementation for **Task 1 – Statistical Analysis of ICS Network Traffic**, a sub-part of the *Adversarial Machine Learning Study Project*.  
The goal is to analyze the **EPIC** dataset (a scaled-down power grid ICS testbed) and extract statistical characteristics from raw **PCAP** files under normal operating conditions.

---

## 🎯 Objective
Perform a **statistical analysis** of the EPIC dataset focusing on:
- Network packet distributions per protocol  
- Inter-arrival time analysis  
- Host communication patterns  
- Cumulative Distribution Function (CDF) plots  
All computations are done **only from raw PCAP files** — no Historian or pre-parsed physical readings are used.

---

## 🧩 Task Description

### **a) General-Purpose Statistics**
- Count total network packets per scenario.  
- Identify application-layer protocols and compute their fractions relative to total packets.  
- Identify transport-layer protocols (e.g., TCP, UDP) and compute their respective fractions.  
- Calculate the fraction of packets for each application-layer protocol per transport-layer protocol.

---

### **b) Packet-Level Statistics**
For packets **containing application data only**:
- Average, standard deviation, and median of **packet length (bytes)** per protocol.  
- Average, standard deviation, and median of **inter-arrival time** between packets exchanged between the same pair of hosts.

---

### **c) Host-Pair Statistics**
- Identify all communicating host pairs in the dataset.  
- Compute average, standard deviation, and median **inter-arrival times** per application-layer protocol type between these host pairs.

---

### **d) Cumulative Distribution Functions (CDFs)**
Plot two **CDFs** per scenario:
1. **Header length (bytes)** per application-layer protocol.  
2. **Application payload length (bytes)** per application-layer protocol.  
> Note: *No pre-existing CDF libraries are allowed — implementation is done from scratch.*

---

## 🧮 Dataset
**EPIC Dataset**  
A simulated power-grid ICS testbed containing eight normal operation scenarios, each lasting 30 minutes.  
Only PCAP files from these scenarios are analyzed.

---

## ⚙️ Tools & Libraries
Recommended Python libraries:
- `scapy` or `pyshark` – packet parsing  
- `pandas`, `numpy` – data processing  
- `matplotlib` – plot generation  
> All statistical computations and CDFs are implemented manually where required.

---

## 📊 Expected Outputs
- Tabular summaries of packet and protocol statistics.  
- CSV/JSON output files for numerical results.  
- CDF plots visualizing header and payload length distributions per protocol per scenario.

---

## 🧠 Learning Outcome
Through this task, students gain experience in:
- Parsing and analyzing raw ICS network traffic  
- Understanding protocol distributions and timing characteristics  
- Implementing statistical computations and visualization from scratch  
- Building the foundation for dataset similarity and synthetic generation in subsequent tasks

---

## 🧑‍💻 Author
**Md Farhan Rahman Anik**  
Brandenburg University of Technology (BTU Cottbus-Senftenberg)

---

## 📅 Deadline
**12 November 2025**

---

## 📎 Reference
EPIC Dataset (Experimental Platform for Internet Contingency):  
[EPIC ICS Dataset SharePoint Link](https://sutdapac-my.sharepoint.com/:f:/g/personal/itrust_sutd_edu_sg/EsEaFko7YGZDtw7lwpruL3oBkmIhPAqBkA3tWgCTXIo6tw?e=W6TBwD)
