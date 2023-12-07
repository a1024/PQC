#include"pqc.h"
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#ifdef _MSC_VER
#include<Windows.h>
#include<intrin.h>
#else
#include<time.h>
#include<x86intrin.h>
#endif
static const char file[]=__FILE__;

CPUInfo cpuinfo;

double time_ms()
{
#ifdef _MSC_VER
	static long long t0=0;
	LARGE_INTEGER li;
	double t;
	QueryPerformanceCounter(&li);
	if(!t0)
		t0=li.QuadPart;
	t=(double)(li.QuadPart-t0);
	QueryPerformanceFrequency(&li);
	t/=(double)li.QuadPart;
	t*=1000;
	return t;
#else
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);//<time.h>
	return t.tv_sec*1000+t.tv_nsec*1e-6;
#endif
}

int pause()
{
	int k;
	printf("Enter 0 to continue: ");
	while(!scanf(" %d", &k));
	return k;
}
int log_error(const char *file, int line, int quit, const char *format, ...)
{
	va_list args;
	printf("\nERROR\n%s(%d):\n", file, line);
	if(format)
	{
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		printf("\n");
	}
	if(quit)
	{
		pause();
		exit(0);
	}
	return 0;
}

int floor_log2(unsigned n)
{
	int logn=0;
	int sh=(n>=1<<16)<<4;	logn+=sh, n>>=sh;
	sh=(n>=1<<8)<<3;	logn+=sh, n>>=sh;
	sh=(n>=1<<4)<<2;	logn+=sh, n>>=sh;
	sh=(n>=1<<2)<<1;	logn+=sh, n>>=sh;
	sh=(n>=1<<1);		logn+=sh;
	return logn;
}
int inv_mod(int x, int n)
{
	int q, r[3]={n, x}, t[3]={0, 1};
	for(;;)//Extended Euclidean Algorithm
	{
		q=r[0]/r[1];//quotient
		r[2]=r[0]%r[1];//new remainder
		if(!r[2])
			break;
		t[2]=t[0]-q*t[1];//new t

		r[0]=r[1], r[1]=r[2];//shift table
		t[0]=t[1], t[1]=t[2];
	}
	if(r[1]!=1)//gcd must be 1 for an inverse to exist
	{
		LOG_ERROR("inv_mod: GCD(%d, %d) = %d\n", x, n, r[1]);
		return 0;
	}
	if(t[1]<0)
		t[1]+=n;
	return t[1];
}
short pow_mod(short x, short e, short q)
{
	int mask[]={1, 0}, product=1;
	if(e<0)
	{
		e=-e;
		mask[1]=inv_mod(x, q);
	}
	else
		mask[1]=x;
	for(;;)
	{
		product*=mask[e&1];
		product%=q;
		e>>=1;
		if(!e)
			return product;
		mask[1]*=mask[1];
		mask[1]%=q;
	}
	return product;
}

void gen_uniform(unsigned char *buf, int len)
{
	for(int k=0;k<len;++k)//TODO use CryptGenRandom
		buf[k]=k+1;
		//buf[k]=rand();
}

static void bitreverse_init(short *bitreverse_table, int logn)
{
	int n=1<<logn;
	for(int k=0;k<n;++k)
	{
		unsigned idx=k;
		idx=(idx>>1&0x5555)|(idx&0x5555)<<1;
		idx=(idx>>2&0x3333)|(idx&0x3333)<<2;
		idx=(idx>>4&0x0F0F)|(idx&0x0F0F)<<4;
		idx=(idx>>8&0x00FF)|(idx&0x00FF)<<8;
		bitreverse_table[k]=idx>>(16-logn);
	}
}
void ntt_init(NTTParams *p, short n, short q, short sqrt_w, short anti_cyclic)
{
	int logn=floor_log2(n);

	p->n=n;//main parameters
	p->q=q;
	p->sqrt_w=sqrt_w;
	p->anti_cyclic=anti_cyclic;

	p->w=sqrt_w*sqrt_w%q;//derived parameters
	p->inv_n=inv_mod(n, q);
	p->beta_q=0x10000%q;
	p->inv_q=inv_mod(q, 0x10000);
	p->beta_stg=pow_mod(p->beta_q, logn, q);//beta^log_2(n), log_2(n)-1 stages + phi/correction
	bitreverse_init(p->bitreverse_table, logn);

	for(int k=0, wk=1;k<n;++k)//initialize NTT roots of unity
	{
		p->roots_fwd[k]=wk;
		p->roots_inv[(2*n-k)%n]=wk;
		wk*=p->w;
		wk%=q;
	}

	{
		int inv_sqrt_w=inv_mod(sqrt_w, q);
		int fwd_f=1, inv_f=inv_mod(n, q);
		for(int k=0;k<n;++k)
		{
			p->phi_fwd[p->bitreverse_table[k]]=fwd_f;//BRP is its own inverse
			p->phi_inv[k]=inv_f;
			fwd_f*=sqrt_w;//bit-reverse permuted phi = BRP{1, phi, ..., phi^(n-1)}		phi=sqrt_w
			inv_f*=inv_sqrt_w;//*{1, -phi^(n-1), ..., -phi}/n mod q
			MOD(fwd_f, fwd_f, q);
			MOD(inv_f, inv_f, q);
		}
		if(fwd_f!=q-1||inv_f*n%q!=q-1)
		{
			LOG_ERROR("Expected all factors to return to -1:\n  fwd %d\n  inv %d\n", fwd_f, inv_f);
			return;
		}
	}
}
static void ntt_impl(short *data, const short *roots, short n, short q, short inv_q)
{
	int k;

	for(k=0;k<n;k+=2)//sum & difference: no modular reduction
	{
		int a0=data[k], a1=data[k+1];
		data[k  ]=a0+a1;
		data[k+1]=a0-a1;
	}
	for(int m=4;m<=n;m*=2)//stage loop
	{
		int rstep=n/m, m_2=m/2;
		for(int k=0;k<n;k+=m)//block loop
		{
			for(int j=0, kr=0;j<m_2;++j, kr+=rstep)//operation loop
			{
				int a0=data[k+j], a1=data[k+j+m_2], b0, b1;
				a1*=roots[kr];
				b0=a0+a1;
				b1=a0-a1;
				b0%=q;
				b1%=q;
				b0+=q&-(b0<0);
				b1+=q&-(b1<0);
				data[k+j]=b0;
				data[k+j+m_2]=b1;
			}
		}
	}
}
void ntt_fwd(NTTParams *p, short *data, int fwd_BRP)
{
	ALIGN(16) short temp[1024];
	if(fwd_BRP)
	{
		for(int k=0;k<p->n;++k)
			temp[k]=data[p->bitreverse_table[k]];
	}
	else
		memcpy(temp, data, p->n*sizeof(short));
	if(p->anti_cyclic)
	{
		for(int k=0;k<p->n;++k)//*{1, phi, ..., phi^(n-1)}
		{
			int val=temp[k];
			val*=p->phi_fwd[k];
			MOD(val, val, p->q);
			temp[k]=val;
		}
	}
	ntt_impl(temp, p->roots_fwd, p->n, p->q, p->inv_q);
	memcpy(data, temp, p->n*sizeof(short));
}
void ntt_inv(NTTParams *p, short *data)
{
	ALIGN(16) short temp[1024];
	for(int k=0;k<p->n;++k)
		temp[k]=data[p->bitreverse_table[k]];
	ntt_impl(temp, p->roots_inv, p->n, p->q, p->inv_q);
	if(p->anti_cyclic)
	{
		for(int k=0;k<p->n;++k)//*{1, -phi^(n-1), ..., -phi}/n mod q
		{
			int val=temp[k];
			val*=p->phi_inv[k];
			MOD(val, val, p->q);
			temp[k]=val;
		}
	}
	memcpy(data, temp, p->n*sizeof(short));
}
void ntt_muladd(NTTParams *p, short *dst, const short *v1, const short *v2)
{
	for(int k=0;k<p->n;++k)
	{
		int val=dst[k]+v1[k]*v2[k];
		MOD(val, val, p->q);//TODO use Montgomery reduction
		dst[k]=val;
	}
}
void vec_add(NTTParams *p, short *dst, const short *v1, const short *v2, int n)
{
	for(int k=0;k<n;++k)
	{
		int val=v1[k]+v2[k];
		MOD(val, val, p->q);//TODO use Montgomery reduction
		dst[k]=val;
	}
}
void vec_sub(NTTParams *p, short *dst, const short *pos, const short *neg, int n)
{
	for(int k=0;k<n;++k)
	{
		int val=pos[k]-neg[k];
		MOD(val, val, p->q);//TODO use Montgomery reduction
		dst[k]=val;
	}
}

static void kyber_uniform_extend(short *dst, int dstcount, unsigned char *src, int srcbytes, short q)
{
	FIPS202_SHAKE128(src, srcbytes, (unsigned char*)dst, dstcount*sizeof(short));
	for(int k=0;k<dstcount;++k)
		*(unsigned short*)(dst+k)%=q;//TODO use Montgomery
}
static void kyber_convert_binomial4(short *v, int n)
{
	const unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	for(int k=0;k<n;++k)
	{
		const unsigned *pm=hamming_masks;
		unsigned a=v[k]&0xFF;//8-bits
		a=(a&*pm)+(a>>1&*pm), ++pm;
		a=(a&*pm)+(a>>2&*pm), ++pm;
		a=(a&*pm)-(a>>4&*pm);
		v[k]=a;
	}
}
static void kyber_compress(short *data, short n, short q, short d)
{
	long long inv_2q=0x400000/(q<<1);
	for(int k=0;k<n;++k)
	{
		int val=data[k];
		val<<=d+1;//round(vk*2^d/q) = floor(((2*vk<<d)+q)/(2*q))
		val+=q;
		val=(int)(val*inv_2q>>22);
		data[k]=(short)val;
	}
}
static void kyber_decompress(short *data, short n, short q, short d)
{
	for(int k=0;k<n;++k)
	{
		int val=data[k];
		val*=q;//round(vk*q/2^d) = (vk*q+(1<<d)/2)>>d
		//val+=1<<(d-1);
		val>>=d;
		data[k]=val;
	}
}
void kyber_gen(short nist_security_level, KyberParams *p, KyberPublicKey *k_pu, KyberPrivateKey *k_pr)
{
	if(!p->p.n)
	{
		//ntt_init(&p->p, 256, 3329, 17, 1);//TODO support the new parameters
		ntt_init(&p->p, 256, 7681, 62, 1);
	}

	switch(nist_security_level)
	{
	case 1:p->security_k=2;break;
	case 3:p->security_k=3;break;
	case 5:p->security_k=4;break;
	default:
		LOG_ERROR("Invalid NIST security level");
		return;
	}
	p->mat_size=p->security_k*p->security_k*p->p.n;
	p->vec_size=p->security_k*p->p.n;
	k_pu->rho=(unsigned char*)_mm_malloc(32, 32);
	MALLOC_CHECK(!k_pu->rho,)
	gen_uniform(k_pu->rho, 32);

	short *A=(short*)_mm_malloc(p->mat_size*sizeof(short), 32);
	MALLOC_CHECK(!A,)
	kyber_uniform_extend(A, p->mat_size, k_pu->rho, 32, p->p.q);

	ALIGN(32) unsigned char sigma[32];
	gen_uniform(sigma, 32);

	short *se_buffer=(short*)_mm_malloc(2*p->vec_size*sizeof(short), 32),
		*s=se_buffer, *e=se_buffer+p->vec_size;
	MALLOC_CHECK(!se_buffer,)
	FIPS202_SHAKE128(sigma, 32, (unsigned char*)se_buffer, 2*p->vec_size*sizeof(short));
	kyber_convert_binomial4(se_buffer, 2*p->vec_size);
	memset(sigma, 0, 32);//security measure

	k_pr->s_ntt=(short*)_mm_malloc(p->vec_size*sizeof(short), 32);
	MALLOC_CHECK(!k_pr->s_ntt,)
	memcpy(k_pr->s_ntt, s, p->vec_size*sizeof(short));
	for(int i=0;i<p->security_k;++i)
		ntt_fwd(&p->p, k_pr->s_ntt+p->p.n*i, 0);//no BRP
	for(int i=0;i<p->security_k;++i)
		ntt_fwd(&p->p, e+p->p.n*i, 0);//no BRP
	
	//t = A s + e
	k_pu->t=(short*)_mm_malloc(p->vec_size*sizeof(short), 32);
	MALLOC_CHECK(!k_pu->t,)
	memcpy(k_pu->t, e, p->vec_size*sizeof(short));
	for(int ky=0;ky<p->security_k;++ky)
	{
		for(int kx=0;kx<p->security_k;++kx)
			ntt_muladd(&p->p, k_pu->t+p->p.n*ky, A+p->p.n*(p->security_k*ky+kx), k_pr->s_ntt+p->p.n*kx);
	}
	for(int i=0;i<p->security_k;++i)
		ntt_inv(&p->p, k_pu->t+p->p.n*i);
	kyber_compress(k_pu->t, p->vec_size, p->p.q, 11);
	
	memset(se_buffer, 0, 2*p->vec_size*sizeof(short));//security measure
	_mm_free(se_buffer);
	_mm_free(A);
}
void kyber_destroy(KyberParams const *p, KyberPublicKey *k_pu, KyberPrivateKey *k_pr, KyberCiphertext *ct)
{
	if(k_pu)
	{
		_mm_free(k_pu->rho);
		_mm_free(k_pu->t);
	}
	if(k_pr)
	{
		if(p)
			memset(k_pr->s_ntt, 0, p->security_k*p->p.n*sizeof(short));
		_mm_free(k_pr->s_ntt);
	}
	if(ct)
	{
		_mm_free(ct->u);
		_mm_free(ct->v);
	}
}
void kyber_enc(KyberParams *p, KyberPublicKey const *k_pu, const void *message, const unsigned char *r_seed, KyberCiphertext *ct)
{
	short *t=(short*)_mm_malloc(p->vec_size*sizeof(short), 32);
	MALLOC_CHECK(!t,)
	memcpy(t, k_pu->t, p->vec_size*sizeof(short));
	kyber_decompress(t, p->vec_size, p->p.q, 11);
	for(int i=0;i<p->security_k;++i)
		ntt_fwd(&p->p, t+p->p.n*i, 1);

	short *A=(short*)_mm_malloc(p->mat_size*sizeof(short), 32);
	MALLOC_CHECK(!A,)
	kyber_uniform_extend(A, p->mat_size, k_pu->rho, 32, p->p.q);
	
	const int re_size=2*p->vec_size+p->p.n;
	short *re_buffer=(short*)_mm_malloc(re_size*sizeof(short), 32),
		*r=re_buffer, *e1=r+p->vec_size, *e2=e1+p->vec_size;
	MALLOC_CHECK(!re_buffer,)
	FIPS202_SHAKE128(r_seed, 32, (unsigned char*)re_buffer, re_size*sizeof(short));
	kyber_convert_binomial4(re_buffer, 2*p->vec_size+p->p.n);
	
	for(int i=0;i<p->security_k;++i)
		ntt_fwd(&p->p, r+p->p.n*i, 0);//no BRP
	
	//u = invNTT(AT r) + e1
	ct->u=(short*)_mm_malloc(p->vec_size*sizeof(short), 32);
	MALLOC_CHECK(!ct->u,)
	memset(ct->u, 0, p->vec_size*sizeof(short));
	for(int ky=0;ky<p->security_k;++ky)
	{
		for(int kx=0;kx<p->security_k;++kx)
			ntt_muladd(&p->p, ct->u+p->p.n*ky, A+p->p.n*(p->security_k*kx+ky), r+p->p.n*kx);
	}
	for(int i=0;i<p->security_k;++i)
		ntt_inv(&p->p, ct->u+p->p.n*i);
	vec_add(&p->p, ct->u, ct->u, e1, p->vec_size);
	kyber_compress(ct->u, p->vec_size, p->p.q, 11);
	
	//v = tT r + e2 + round(q/2)*m
	ct->v=(short*)_mm_malloc(p->p.n*sizeof(short), 32);
	MALLOC_CHECK(!ct->v,)
	memset(ct->v, 0, p->p.n*sizeof(short));
	for(int i=0;i<p->security_k;++i)
		ntt_muladd(&p->p, ct->v, t+p->p.n*i, r+p->p.n*i);
	ntt_inv(&p->p, ct->v);
	short half=(p->p.q>>1)+1;
	for(int i=0;i<p->p.n;++i)
	{
		int val=ct->v[i];
		val+=e2[i];
		val+=half&-(((unsigned char*)message)[i>>3]>>(i&7)&1);
		MOD(val, val, p->p.q);
		ct->v[i]=val;
	}
	kyber_compress(ct->v, p->p.n, p->p.q, 3);
	
	//cleanup
	memset(re_buffer, 0, re_size*sizeof(short));
	_mm_free(re_buffer);
	_mm_free(A);
}
void kyber_dec(KyberParams *p, KyberPrivateKey const *k_pr, KyberCiphertext const *ct, void *dst)
{
	short *u_ntt=(short*)_mm_malloc(p->vec_size*sizeof(short), 32);
	MALLOC_CHECK(!u_ntt,);
	memcpy(u_ntt, ct->u, p->vec_size*sizeof(short));
	kyber_decompress(u_ntt, p->vec_size, p->p.q, 11);
	for(int i=0;i<p->security_k;++i)
		ntt_fwd(&p->p, u_ntt+p->p.n*i, 1);

	short *v_ntt=(short*)_mm_malloc(p->p.n*sizeof(short), 32);
	MALLOC_CHECK(!v_ntt,);
	memcpy(v_ntt, ct->v, p->p.n*sizeof(short));
	kyber_decompress(v_ntt, p->p.n, p->p.q, 3);
	ntt_fwd(&p->p, v_ntt, 1);
	
	//m = v - sT u
	short *m2=(short*)_mm_malloc(p->p.n*sizeof(short), 32);
	MALLOC_CHECK(!m2,);
	memset(m2, 0, p->p.n*sizeof(short));
	for(int i=0;i<p->security_k;++i)
		ntt_muladd(&p->p, m2, k_pr->s_ntt+p->p.n*i, u_ntt+p->p.n*i);
	vec_sub(&p->p, m2, v_ntt, m2, p->p.n);
	ntt_inv(&p->p, m2);
	
	unsigned char *ptr=(unsigned char*)dst;
	memset(dst, 0, p->p.n>>3);
	for(int i=0, start=p->p.q/4, end=p->p.q*3/4;i<p->p.n;++i)
	{
		int bit=(m2[i]>start)&(m2[i]<end);
		ptr[i>>3]|=bit<<(i&7);
	}
}

KyberParams params;
int main(int argc, char **argv)
{
	printf("Post-Quantum Crypto Benchmark\n");
	get_cpuinfo(&cpuinfo);
	print_cpuinfo(&cpuinfo);

	KyberPublicKey kpu;
	KyberPrivateKey kpr;
	kyber_gen(3, &params, &kpu, &kpr);

	const char message[]="12345678901234567890123456789012";
	printf("%s\n", message);
	KyberCiphertext ct;
	char seed[32];
	gen_uniform(seed, 32);
	kyber_enc(&params, &kpu, message, seed, &ct);

	char m2[33]={0};
	kyber_dec(&params, &kpr, &ct, m2);

	printf("%s\n", m2);

	kyber_destroy(&params, &kpu, &kpr, &ct);

	printf("Done.\n");
	pause();
	return 0;
}
