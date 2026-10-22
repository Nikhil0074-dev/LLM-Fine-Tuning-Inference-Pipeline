"""Create a synthetic customer-support dataset (JSONL). Run: python scripts/make_demo_dataset.py out.jsonl

WARNING: validation prompts are paraphrases of training prompts, so scores on this data are
optimistic. Use it to test the pipeline, not to judge model quality.
"""
import json
import sys

INTENTS = {
    "To reset your password, open the sign-in page, choose Forgot password, and follow the link we email you.": [
        "how do I reset my password", "I forgot my password what now", "cannot log in need a new password"],
    "You can download invoices from Billing > History. Pick a month and choose Download PDF.": [
        "where can I download my invoice", "I need a receipt for last month", "how to get billing history pdf"],
    "To cancel, go to Settings > Plan and choose Cancel subscription. Access stays active until the period ends.": [
        "how do I cancel my subscription", "I want to stop my plan", "cancel my account please"],
    "We support Visa, Mastercard and bank transfer. Update your method in Billing > Payment methods.": [
        "which payment methods do you accept", "can I pay by bank transfer", "how do I change my card"],
    "Open Settings > Team, click Invite, and enter your colleague's work address. They get a link right away.": [
        "how do I add a teammate", "invite a new user to my workspace", "adding people to the team"],
    "Our support team answers within 24 hours on weekdays. Use the Help button in the app to open a ticket.": [
        "how fast does support reply", "how do I contact support", "I need to talk to a human"],
    "Go to Settings > Security and switch on Two-factor authentication. Scan the QR code with an authenticator app.": [
        "how do I turn on two factor", "enable 2fa on my account", "make my login more secure"],
    "Use Settings > Data > Export. We prepare a ZIP file and email you a download link within one hour.": [
        "how can I export my data", "download all my information", "get a backup of my workspace"],
    "Refunds are available within 14 days of purchase. Write to the billing team with your order number.": [
        "can I get a refund", "I was charged by mistake", "how do refunds work"],
    "Check status.example.com for live incidents. If everything is green, clear your cache and try again.": [
        "the app is not loading", "service seems down", "pages keep failing to open"],
}
WRAP = ["", "hi, ", "hello, ", "quick question: ", "please help, ", "hey team, ", "one more thing: ", "excuse me, ",
        "good morning, ", "sorry to bother, ", "urgent: ", "thanks in advance, ", "help needed - ", "question - ", "so, "]


def main(path: str) -> None:
    rows = []
    for answer, questions in INTENTS.items():
        for q in questions:
            for w in WRAP[:5]:
                rows.append({"prompt": (w + q).strip().capitalize() + "?", "response": answer})
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample_data/support.jsonl")
