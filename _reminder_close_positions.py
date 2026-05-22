import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.notifications.slack_notifier import send_message

send_message(
    ":bell: *REMINDER — Close all positions before FOMC*\n"
    "> Close SPY 714/713 P (exp 5/8)\n"
    "> Close SPY 717/715 P (exp 5/11)\n"
    "> Close SPY 714/707 P (exp 5/11)\n"
    "> Fed announcement at 2 PM ET — get out by 10 AM to be safe\n"
    "> Then run: `python run_options_paper.py settle`"
)
print("Reminder sent to Slack.")
