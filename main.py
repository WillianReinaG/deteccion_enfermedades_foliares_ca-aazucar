from app.agent.agent import SugarCaneAgent

if __name__ == "__main__":
    agent = SugarCaneAgent()
    print("SugarCane AI Agent listo. Escribe 'salir' para terminar.")
    while True:
        q = input("Usuario: ").strip()
        if q.lower() in {"salir", "exit", "quit"}:
            break
        print("Agente:", agent.answer(q))
