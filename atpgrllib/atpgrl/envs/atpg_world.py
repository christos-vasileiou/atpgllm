import gym
from gym import spaces
import fcntl
import numpy as np
import os
import subprocess

pattern_simulation_template = """# Open the file for reading
set file [open [lindex {pattern_file} 0] "r"]

# Initialize a pattern count variable
set pattern_count 0

# Read the file line by line
while {[gets $file line] != -1} {
  # Check for the "Pattern" pattern
  if {[string match "*Pattern*" $line]} {
    incr pattern_count
    continue
  }
  # Check for the "Time 0: force_all_pis =" pattern
  if {[string match "*Time 0: force_all_pis =*" $line]} {
    # Split the line into fields based on spaces
    set fields [split $line " "]
    # Loop through the fields starting from the 6th field
    for {set i 5} {$i < [llength $fields]} {incr i} {
      # Print the field and add a space (or newline for the last field)
      set arg [lindex $fields $i]
      if {$i == [expr {[llength $fields] - 1}]} {
        puts $arg
      } else {
        puts -nonewline "$arg "
      }
    }
  }
}
# Close the file
close $file
"""

data = {'pattern_file': 0,
        
}

class ATPGWorldEnv(gym.Env):
  action_to_tcl = {"reset": "drc -force\nread_netlist -delete\n",
                    "atpg": "",
                    "": "",
                    }
  
  def __init__(self, batch_size=4):
    super(ATPGWorldEnv, self).__init__()
    # Define action and observation space

    # Initialize external tool
    self.tmax_processes = [subprocess.Popen("source /proj/cad/startup/profile.synopsys_2018 && tmax -shell",
                                         shell=True,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         executable='/bin/bash', # Specify the shell
                                         text=True) for _ in range(batch_size)]
    for tp in self.tmax_processes:
      tp.stdin.write(self._action_to_tcl('reset'))
      tp.stdin.flush()
      # set the stdout to non-blocking
      fd_o = tp.stdout.fileno()
      fl_o = fcntl.fcntl(fd_o, fcntl.F_GETFL)
      fcntl.fcntl(fd_o, fcntl.F_SETFL, fl_o | os.O_NONBLOCK)
      # set the stderr to non-blocking
      fd_e = tp.stderr.fileno()
      fl_e = fcntl.fcntl(fd_e, fcntl.F_GETFL)
      fcntl.fcntl(fd_e, fcntl.F_SETFL, fl_e | os.O_NONBLOCK)
    
    # create a temporary folder to keep track of info
    self.info_path = './temp'
    os.makedirs(self.info_path, exist_ok=True)
    
    ### Print PIDs
    self.print_subprocess_pids()
    
  def read_output(std):
    """
    std: should _process.stdout or _process.stderr
    """
    outputs = []
    while True:
      try:
        output = std.readline()
        if not output:
          print('Break')
          break
        print(output)
        outputs.append(output)
      except IOError:
        print("IOError")
        pass
    return outputs

  def _get_obs(self):
    pass

  def _get_info(self):
    pass

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.tmax_command.stdin.write(self._action_to_tcl("reset"))
    self.tmax_command.stdin.flush()

  def _action_to_tcl(self, action):
    # Convert an action to TCL command(s)
    tcl_command = self.action_to_tcl[action]
    return tcl_command

  def step(self, action):
    # Convert the action into TCL command(s)
    tcl_command = self._action_to_tcl(action)
    # Send the command to TetraxMax and get the response
    self.tmax_process.stdin.write(tcl_command+'\n')
    self.tmax_process.stdin.flush()
    output = self.tmax_process.stdout.readline()
    # Process the output and update the environment's state
    # ...
    
    info = output

    return self._get_observation(), reward, terminated, done, info

  def print_subprocess_pids(self):
    for idx, proc in enumerate(self.tmax_processes):
      print(f"Subprocess {idx} PID: {proc.pid}")

  def render(self):
    pass

  def _render_frame(self):
    pass

  def close(self):
    for tp in self.tmax_process:
      self.tmax_process.stdin.write('quit\n')
      self.tmax_process.stdin.flush()
      output = self.tmax_process.stdout.readline()
    
