"""
Payload serialization and cryptographic framing.

This module converts messages/files into bitstreams using a fixed
binary container: MAGIC + VERSION + SALT + NONCE + AES-GCM ciphertext.
Keys are derived with PBKDF2-HMAC-SHA256, and payloads are compressed
before encryption. It also provides the inverse decoding pipeline.
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

MAGIC = b"STG1"
VERSION = 1
SALT_SIZE = 16
NONCE_SIZE = 12
PBKDF2_ITERS = 200_000
KEY_SIZE = 32


def _derive_key(password: str, salt: bytes) -> bytes:
	if not password:
		raise ValueError("password must not be empty")
	kdf = PBKDF2HMAC(
		algorithm=hashes.SHA256(),
		length=KEY_SIZE,
		salt=salt,
		iterations=PBKDF2_ITERS,
	)
	return kdf.derive(password.encode("utf-8"))


def _encrypt_payload(data: bytes, password: str) -> bytes:
	salt = os.urandom(SALT_SIZE)
	nonce = os.urandom(NONCE_SIZE)
	key = _derive_key(password, salt)
	aesgcm = AESGCM(key)

	compressed = zlib.compress(data, level=9)
	plaintext = struct.pack(">I", len(data)) + compressed
	ciphertext = aesgcm.encrypt(nonce, plaintext, None)

	return MAGIC + bytes([VERSION]) + salt + nonce + ciphertext


def _decrypt_payload(blob: bytes, password: str) -> bytes:
	header_len = len(MAGIC) + 1 + SALT_SIZE + NONCE_SIZE
	if len(blob) < header_len:
		raise ValueError("payload is too small")

	magic = blob[: len(MAGIC)]
	if magic != MAGIC:
		raise ValueError("payload magic mismatch")

	version = blob[len(MAGIC)]
	if version != VERSION:
		raise ValueError("unsupported payload version")

	offset = len(MAGIC) + 1
	salt = blob[offset : offset + SALT_SIZE]
	offset += SALT_SIZE
	nonce = blob[offset : offset + NONCE_SIZE]
	offset += NONCE_SIZE
	ciphertext = blob[offset:]

	key = _derive_key(password, salt)
	aesgcm = AESGCM(key)
	plaintext = aesgcm.decrypt(nonce, ciphertext, None)

	if len(plaintext) < 4:
		raise ValueError("decrypted payload is too small")

	expected_len = struct.unpack(">I", plaintext[:4])[0]
	compressed = plaintext[4:]
	data = zlib.decompress(compressed)

	if len(data) != expected_len:
		raise ValueError("payload length mismatch")

	return data


def bytes_to_bits(data: bytes) -> np.ndarray:
	if not data:
		raise ValueError("data must not be empty")
	arr = np.frombuffer(data, dtype=np.uint8)
	return np.unpackbits(arr, bitorder="big")


def bits_to_bytes(bits: np.ndarray) -> bytes:
	bits_arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
	if bits_arr.size == 0:
		raise ValueError("bits must not be empty")
	if bits_arr.size % 8 != 0:
		raise ValueError("bit length must be a multiple of 8")
	packed = np.packbits(bits_arr, bitorder="big")
	return packed.tobytes()


def encode_message(message: str, password: str) -> np.ndarray:
	data = message.encode("utf-8")
	payload = _encrypt_payload(data, password)
	return bytes_to_bits(payload)


def encode_file(file_path: str, password: str) -> np.ndarray:
	with open(file_path, "rb") as handle:
		data = handle.read()
	payload = _encrypt_payload(data, password)
	return bytes_to_bits(payload)


def decode_bits(bits: np.ndarray, password: str) -> bytes:
	payload = bits_to_bytes(bits)
	return _decrypt_payload(payload, password)
