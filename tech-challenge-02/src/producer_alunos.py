"""
Produtor Kafka - Simulação de eventos de avaliação de alunos
Tech Challenge Fase 2 - Pipeline Híbrida de Alfabetização

Simula a chegada de novas medições de proficiência (Saeb) para alunos,
publicando eventos JSON em um tópico do Confluent Cloud.

Configuração via variáveis de ambiente (não commitar credenciais):
    export CONFLUENT_BOOTSTRAP_SERVERS="<host>.confluent.cloud:9092"
    export CONFLUENT_API_KEY="<sua_api_key>"
    export CONFLUENT_API_SECRET="<seu_api_secret>"
    export KAFKA_TOPIC="alunos-eventos"

Instalação:
    pip install confluent-kafka --break-system-packages

Uso:
    python producer_alunos.py --intervalo 3 --quantidade 100
"""

import argparse
import json
import os
import random
import sys
import time
import uuid

from confluent_kafka import Producer

# Ponto de corte definido pela Pesquisa Alfabetiza Brasil (INEP, 2023)
PONTO_CORTE_ALFABETIZACAO = 743

# Amostra de códigos de município (IBGE) para simulação
IDS_MUNICIPIO_SIMULADOS = [
    "3550308",  # São Paulo - SP
    "3304557",  # Rio de Janeiro - RJ
    "2304400",  # Fortaleza - CE
    "2611606",  # Recife - PE
    "1302603",  # Manaus - AM
    "5300108",  # Brasília - DF
    "4106902",  # Curitiba - PR
    "2927408",  # Salvador - BA
]

ANOS_SIMULADOS = [2023, 2024]
CADERNOS_SIMULADOS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
SERIES_SIMULADAS = ["2"]  # 2º ano do ensino fundamental
REDES_SIMULADAS = ["2", "3"]
PRESENCA_SIMULADA = ["0", "1"]
PREENCHIMENTO_SIMULADO = ["0", "1"]


def carregar_config() -> dict:
    """Lê as credenciais do Confluent Cloud a partir de variáveis de ambiente."""
    obrigatorias = [
        "CONFLUENT_BOOTSTRAP_SERVERS",
        "CONFLUENT_API_KEY",
        "CONFLUENT_API_SECRET",
    ]
    faltando = [v for v in obrigatorias if not os.environ.get(v)]
    if faltando:
        print(f"Erro: variáveis de ambiente ausentes: {', '.join(faltando)}")
        sys.exit(1)

    return {
        "bootstrap.servers": os.environ["CONFLUENT_BOOTSTRAP_SERVERS"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": os.environ["CONFLUENT_API_KEY"],
        "sasl.password": os.environ["CONFLUENT_API_SECRET"],
    }


def gerar_evento_aluno() -> dict:
    """Gera um evento simulado de avaliação de proficiência de um aluno,
    seguindo o schema: ano, id_municipio, id_escola, id_aluno, caderno,
    serie, rede, presenca, preenchimento_caderno, alfabetizado,
    proficiencia, peso_aluno.
    """
    presenca = random.choices(PRESENCA_SIMULADA, weights=[0.92, 0.08])[0]

    if presenca == "0":
        preenchimento_caderno = random.choices(
            PREENCHIMENTO_SIMULADO, weights=[0.85, 0.15]
        )[0]
    else:
        preenchimento_caderno = "1"

    # Só há proficiência/alfabetização válida se o aluno esteve presente
    # e preencheu o caderno adequadamente (mesma lógica do Saeb real)
    if presenca == "0" and preenchimento_caderno == "0":
        proficiencia = round(random.gauss(mu=720, sigma=90), 1)
        proficiencia = max(0.0, min(1000.0, proficiencia))
        alfabetizado = "1" if proficiencia >= PONTO_CORTE_ALFABETIZACAO else "0"
    else:
        proficiencia = None
        alfabetizado = "0"

    return {
        "ano": random.choice(ANOS_SIMULADOS),
        "id_municipio": random.choice(IDS_MUNICIPIO_SIMULADOS),
        "id_escola": f"{random.randint(10000000, 99999999)}",
        "id_aluno": "TEST",
        # "id_aluno": f"{random.randint(53027701, 99999999)}",
        "caderno": random.choice(CADERNOS_SIMULADOS),
        "serie": random.choice(SERIES_SIMULADAS),
        "rede": random.choice(REDES_SIMULADAS),
        "presenca": presenca,
        "preenchimento_caderno": preenchimento_caderno,
        "alfabetizado": alfabetizado,
        "proficiencia": proficiencia,
        "peso_aluno": round(random.uniform(0.5, 2.5), 4),
    }


def callback_entrega(erro, msg):
    """Callback chamado de forma assíncrona após tentativa de entrega da mensagem."""
    if erro is not None:
        print(f"Falha ao entregar mensagem: {erro}")
    else:
        print(
            f"Entregue -> tópico={msg.topic()} partição={msg.partition()} "
            f"offset={msg.offset()}"
        )


def main():
    parser = argparse.ArgumentParser(description="Simulador de eventos de alunos para Kafka")
    parser.add_argument("--intervalo", type=float, default=2.0, help="Segundos entre eventos (padrão: 2)")
    parser.add_argument("--quantidade", type=int, default=0, help="Nº de eventos a enviar (0 = infinito)")
    args = parser.parse_args()

    topico = os.environ.get("KAFKA_TOPIC", "alunos-eventos")
    config = carregar_config()
    produtor = Producer(config)

    enviados = 0
    print(f"Iniciando produtor -> tópico '{topico}' | intervalo={args.intervalo}s")

    try:
        while args.quantidade == 0 or enviados < args.quantidade:
            evento = gerar_evento_aluno()
            produtor.produce(
                topic=topico,
                key=evento["id_municipio"],
                value=json.dumps(evento),
                callback=callback_entrega,
            )
            produtor.poll(0)  # dispara callbacks pendentes sem bloquear
            enviados += 1
            time.sleep(args.intervalo)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        print("Aguardando entrega das mensagens pendentes...")
        produtor.flush(10)
        print(f"Total de eventos enviados: {enviados}")


if __name__ == "__main__":
    main()