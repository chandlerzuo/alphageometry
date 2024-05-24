#!/usr/bin/env bash

# NOTE: this script is not thread-safe since multiple processes write to the same file!

ip_filename="./logs/live_servers.txt"

# Function to get the IP address and append it to a file
get_and_append_ip() {
  ip_address=$(hostname -I | awk '{print $1}')

  if [ -n "$ip_address" ]; then
    echo "$ip_address" >> "$ip_filename"
    echo "IP address $ip_address written to $ip_filename"
  else
    echo "Error: Unable to obtain IP address"
    exit 1
  fi
}

# Function to check if a server is reachable
is_server_reachable() {
  ping -c 1 "$1" > /dev/null
}

# Read the list of previous servers from the file
while read -r server_ip; do
  # Check if the server is reachable
  if is_server_reachable "$server_ip"; then
    echo "Server at $server_ip is reachable."
  else
    echo "Server at $server_ip is not reachable. Removing from the list."
    sed -i "/$server_ip/d" "$ip_filename"
  fi
done < "$ip_filename"

# get and write IP address of this machine
get_and_append_ip

# # initialize conda itself
# eval "$(conda shell.bash hook)"

# # Activate your Conda environment
# conda activate llama_env

# # There is a problem. If two jobs start at the same machine, we would have trouble
# # with the port. Also we have to verify properly if the lmql is running!
# # Start the server
# lmql serve-model gpt2-medium --port 3630 --host 0.0.0.0 --cuda &
