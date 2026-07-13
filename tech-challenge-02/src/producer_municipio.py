"""
Produtor Kafka - Simulação de atualizações de indicadores municipais
Tech Challenge Fase 2 - Pipeline Híbrida de Alfabetização

Simula a chegada de novas medições de indicadores de alfabetização por
município (taxa, média de proficiência, proporção por nível), publicando
eventos JSON em um tópico do Confluent Cloud.

Configuração via variáveis de ambiente:
    export CONFLUENT_BOOTSTRAP_SERVERS="<host>.confluent.cloud:9092"
    export CONFLUENT_API_KEY="<sua_api_key>"
    export CONFLUENT_API_SECRET="<seu_api_secret>"
    export KAFKA_TOPIC="municipio-eventos"

Instalação:
    pip install confluent-kafka --break-system-packages

Uso:
    python producer_municipio.py --intervalo 3 --quantidade 20
"""

import argparse
import json
import os
import random
import sys
import time

from confluent_kafka import Producer

IDS_MUNICIPIO_SIMULADOS = [
    "3550308",  # São Paulo - SP
    "3304557",  # Rio de Janeiro - RJ
    "2304400",  # Fortaleza - CE
    "2611606",  # Recife - PE
    "1302603",  # Manaus - AM
    "5300108",  # Brasília - DF
    "4106902",  # Curitiba - PR
    "2927408",  # Salvador - BA
    "3106200",  # Belo Horizonte - MG
    "1100205",  # Porto Velho - RO
]

ANOS_SIMULADOS = [2023, 2024]
SERIES_SIMULADAS = ["2"]
REDES_SIMULADAS = ["2", "3"]


def _gerar_proporcoes_niveis() -> dict:
    """Gera proporções aleatórias por nível de aprendizagem (soma ≈ 1.0)."""
    pesos = [random.random() for _ in range(9)]
    total = sum(pesos)
    proporcoes = [round(p / total, 4) for p in pesos]
    proporcoes[-1] = round(1.0 - sum(proporcoes[:-1]), 4)
    return {f"proporcao_aluno_nivel_{i}": proporcoes[i] for i in range(9)}


def carregar_config() -> dict:
    obrigatorias = ["CONFLUENT_BOOTSTRAP_SERVERS", "CONFLUENT_API_KEY", "CONFLUENT_API_SECRET"]
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


def gerar_evento_municipio() -> dict:
    taxa = round(max(0.0, min(100.0, random.gauss(mu=75.0, sigma=12.0))), 2)
    media = round(max(500.0, min(950.0, random.gauss(mu=745.0, sigma=60.0))), 2)

    evento = {
        "ano": random.choice(ANOS_SIMULADOS),
        # "id_municipio": random.choice(IDS_MUNICIPIO_SIMULADOS),
        "id_municipio": 'TEST',
        "serie": random.choice(SERIES_SIMULADAS),
        "rede": random.choice(REDES_SIMULADAS),
        "taxa_alfabetizacao": taxa,
        "media_portugues": media,
    }
    evento.update(_gerar_proporcoes_niveis())
    return evento


def callback_entrega(erro, msg):
    if erro is not None:
        print(f"Falha ao entregar mensagem: {erro}")
    else:
        print(
            f"Entregue -> tópico={msg.topic()} partição={msg.partition()} "
            f"offset={msg.offset()}"
        )


def main():
    parser = argparse.ArgumentParser(description="Simulador de indicadores municipais para Kafka")
    parser.add_argument("--intervalo", type=float, default=3.0, help="Segundos entre eventos (padrão: 3)")
    parser.add_argument("--quantidade", type=int, default=0, help="Nº de eventos a enviar (0 = infinito)")
    args = parser.parse_args()

    topico = os.environ.get("KAFKA_TOPIC", "municipio-eventos")
    config = carregar_config()
    produtor = Producer(config)

    enviados = 0
    print(f"Iniciando produtor -> tópico '{topico}' | intervalo={args.intervalo}s")

    try:
        while args.quantidade == 0 or enviados < args.quantidade:
            evento = gerar_evento_municipio()
            produtor.produce(
                topic=topico,
                key=evento["id_municipio"],
                value=json.dumps(evento),
                callback=callback_entrega,
            )
            produtor.poll(0)
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
