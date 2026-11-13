from search import search_prompt

EXIT_COMMANDS = {"sair", "exit"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAté logo!")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Até logo!")
            break

        try:
            answer = chain(question)
        except KeyboardInterrupt:
            print("\nPergunta cancelada.\n")
            continue
        except Exception as e:
            print(f"Erro ao responder a pergunta: {e}\n")
            continue

        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
