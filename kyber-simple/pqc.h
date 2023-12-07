#pragma once
#ifndef PQC_H
#define PQC_H
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include<stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
#ifdef _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __GNUC__
#define ALIGN(X) __attribute((aligned(X)))
#endif
	
typedef struct CPUInfoStruct
{
	char vendor[32], brand[64];
	char
		mmx,
		sse,
		sse2,
		sse3,
		ssse3,
		sse4_1,
		sse4_2,
		fma,
		aes,
		sha,
		avx,
		avx2,
		avx512F,
		avx512PF,
		avx512ER,
		avx512CD,
		f16c,
		rdrand,
		rdseed;
} CPUInfo;
void get_cpuinfo(CPUInfo *info);
void print_cpuinfo(CPUInfo *info);
extern CPUInfo cpuinfo;

int pause();
int log_error(const char *file, int line, int quit, const char *format, ...);
#define LOG_ERROR(X, ...) log_error(file, __LINE__, 1, X, ##__VA_ARGS__)
#define LOG_WARN(X, ...)  log_error(file, __LINE__, 0, X, ##__VA_ARGS__)
#define MALLOC_CHECK(COND, RET)\
	if(COND)\
	{\
		LOG_ERROR("Allocation error\n");\
		return RET;\
	}

#define MOD(DST, X, N) DST=(X)%(N), DST+=(N)&-(DST<0)
int floor_log2(unsigned n);
int inv_mod(int x, int n);
short pow_mod(short x, short e, short q);


void gen_uniform(unsigned char *buf, int len);

//SHA-3
void Keccak(unsigned r, unsigned c, const unsigned char *in, unsigned long long inLen, unsigned char sfx, unsigned char *out, unsigned long long outLen);
#define FIPS202_SHA3_224(IN, INLEN, OUT) Keccak(1152,  448, IN, INLEN, 0x06, OUT, 28)
#define FIPS202_SHA3_256(IN, INLEN, OUT) Keccak(1088,  512, IN, INLEN, 0x06, OUT, 32)
#define FIPS202_SHA3_384(IN, INLEN, OUT) Keccak( 832,  768, IN, INLEN, 0x06, OUT, 48)
#define FIPS202_SHA3_512(IN, INLEN, OUT) Keccak( 576, 1024, IN, INLEN, 0x06, OUT, 64)
#define FIPS202_SHAKE128(IN, INLEN, OUT, OUTLEN) Keccak(1344, 256, IN, INLEN, 0x1F, OUT, OUTLEN)
#define FIPS202_SHAKE256(IN, INLEN, OUT, OUTLEN) Keccak(1088, 512, IN, INLEN, 0x1F, OUT, OUTLEN)


#define N_MAX 1024
typedef struct NTTParamsStruct
{
	short n, q, sqrt_w, anti_cyclic;//main parameters
	short w, inv_n, beta_q, inv_q, beta_stg;//derived parameters
	short bitreverse_table[N_MAX];
	short roots_fwd[N_MAX], roots_inv[N_MAX];
	short phi_fwd[N_MAX], phi_inv[N_MAX];
} NTTParams;
typedef struct KyberParamsStruct
{
	NTTParams p;
	short security_k, mat_size, vec_size;
} KyberParams;
typedef struct KyberPrivateKeyStruct
{
	short *s_ntt;//768 *13bit
} KyberPrivateKey;
typedef struct KyberPublicKeyStruct
{
	unsigned char *rho;	//256bit = 32 bytes
	short *t;		//768 *11bit -> 528 shorts = 1056 bytes
} KyberPublicKey;
typedef struct KyberCiphertextStruct
{
	short *u,	//768 *11bit -> 528 shorts = 1056 bytes
		*v;	//256 *3bit -> 48 shorts = 96 bytes
} KyberCiphertext;
typedef struct Kyber_KE_CiphertextStruct
{
	unsigned char *u,	//1056 bytes
		*v,		//96 bytes
		*d;		//32 bytes
} Kyber_KE_Ciphertext;
void kyber_gen(short nist_security_level, KyberParams *p, KyberPublicKey *k_pu, KyberPrivateKey *k_pr);
void kyber_destroy(KyberParams const *p, KyberPublicKey *k_pu, KyberPrivateKey *k_pr, KyberCiphertext *ct);
void kyber_enc(KyberParams *p, KyberPublicKey const *k_pu, const void *message, const unsigned char *r_seed, KyberCiphertext *ct);
void kyber_dec(KyberParams *p, KyberPrivateKey const *k_pr, KyberCiphertext const *ct, void *dst);

#ifdef __cplusplus
}
#endif
#endif
