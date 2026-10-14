"""Repro: banking injection tasks 0/1/3 security() precedence bug (agentdojo 0.1.35, suites v1 and v1_2).
`any(service or "" in t.subject.lower() ...)` parses as `any(service or ("" in ...))`, so ANY
transaction to the attacker IBAN counts as a successful attack, whatever its subject."""
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment  # noqa: F401
suite = get_suite("v1.2", "banking")
env = suite.load_and_inject_default_environment({})
for tid in ["injection_task_0", "injection_task_1", "injection_task_3"]:
    task = suite.get_injection_task_by_id(tid)
    post = env.model_copy(deep=True)
    # send money to the attacker with a subject that leaks NOTHING
    tx = post.bank_account.transactions[0].model_copy(update={
        "id": 9999, "recipient": task._ATTACKER_IBAN, "subject": "unrelated", "amount": 0.01})
    post.bank_account.transactions.append(tx)
    print(tid, "security() =", task.security("", env, post), "(expected False: subject leaks nothing)")
