# Premium Client Churn Prediction

## Project Installation

### Local

Create virtual environment

```bash
python -m venv .venv
```

Activate venv

```bash
# Windows
.venv\Scripts\activate

# Linux
source .venv/Scripts/activate
```

Install requirements.txt

```
pip install -r requirements.txt
```

Execute main

```
python -u main.py
```

### Virtual Machine

First! Check if python and venv are installed
```bash
python3 --version
sudo apt install -y python3.12-venv
```

THEN, clone the repo
```bash
git clone https://github.com/JessicaAquino/client-retention-analytics.git
```

Inside the folder...
```bash
cd client-retention-analytics
```

Create and activate the environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements (for the VM)
```bash
pip install -r vm_requirements.txt
```

Now! We execute the main.py as a background process. The output will be saved on the vm_execution.log.
```bash
# With this, we send the process to background
nohup python3 -u main.py > vm_execution.log 2>&1 &
```

### Bonus

To check all the executed commands previously.
```bash
history
```

To execute an specific command (with a specific history number)
```bash
!15
```

To check all processes that are being processed
```bash
ps aux

# To get only info related to a specific PID (Process ID)
ps aux | grep 1234
```

To terminate a process execution for a PID.
```bash
kill 1234
```

To check how our process is working...
```bash
tail -f vm_execution.log
```

To edit a file in our console.

```bash
nano my_file.txt
```

* `Ctrl + O` Saving a file
* `Ctrl + X` Exit

That's all folks!
