#his agent perceives its location (A or B) and status (Dirty or Clean) and acts accordingly.

class VacuumAgent:
    def __init__(self):
        # Environment: Location A and B status (1=Dirty, 0=Clean)
        self.location_status = {'A': 1, 'B': 1}
        self.current_pos = 'A'

    def sense_and_act(self):
        for _ in range(2):  # Iterate through locations
            status = self.location_status[self.current_pos]
            print(f"Location {self.current_pos} is {'Dirty' if status else 'Clean'}")

            if status == 1:
                print(f"Action: Suck (Cleaning {self.current_pos}...)")
                self.location_status[self.current_pos] = 0
            else:
                print(f"Action: None")

            # Move to the other location
            self.current_pos = 'B' if self.current_pos == 'A' else 'A'
            print(f"Action: Move to {self.current_pos}")

# Execution
agent = VacuumAgent()
agent.sense_and_act()