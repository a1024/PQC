#undef		UNICODE
#include	<Windows.h>
#include	<Wincrypt.h>
#include	<iostream>
#include	<string>
#include	<vector>
#include	<map>
#include	<set>
#include	<sstream>
#include	<fstream>
#include	<math.h>
#include	<conio.h>
#define		IA_32	-1
#define		MMX		0
#define		SSE		1
#define		SSE2	2
#define		SSE3	3
#define		SSE3S	4
#define		SSE4_1	5
#define		SSE4_2	6
#define		AVX		7
#define		AVX2	8
#define		AVX_512	9

//	#define		PROCESSOR_ARCH	AVX2
	#define		PROCESSOR_ARCH	SSE3
//	#define		PROCESSOR_ARCH	IA_32

//	#define		PROCESSOR_ARCH_AES_NI

	#define		NTT_MULTIPLICATION
//	#define		NTT_TEST

	#define		IA32_USE_MONTGOMERY_REDUCTION

//	#define		XOF_USE_KECCAK_TINY
//	#define		XOF_USE_TWEETFIPS202
	#define		XOF_USE_DRBG_AES128

	#define		R5_USE_IDX//
//	#define		PROFILER

#ifdef XOF_USE_KECCAK_TINY
#include	"keccak-tiny.h"
#endif
#if PROCESSOR_ARCH>=AVX		//i5 2410M, i7 6800K
#include	<immintrin.h>
#elif PROCESSOR_ARCH==SSE4_2//i5 430M
#include	<nmmintrin.h>
#elif PROCESSOR_ARCH==SSE4_1
#include	<smmintrin.h>
#elif PROCESSOR_ARCH==SSE3S	//c2d u7700
#include	<tmmintrin.h>
#elif PROCESSOR_ARCH==SSE3
#include	<pmmintrin.h>
#elif PROCESSOR_ARCH==SSE2
#include	<emmintrin.h>
#elif PROCESSOR_ARCH==SSE
#include	<xmmintrin.h>
#elif PROCESSOR_ARCH==MMX
#include	<mmintrin.h>
#endif
#ifdef PROCESSOR_ARCH_AES_NI
#include	<wmmintrin.h>
#endif
struct NTT_params_AVX2
{
	bool anti_cyclic;
	short n, q, w, sqrt_w, n_1, beta_q, q_1, sbar_m;
	__m256i *m_phi, *m_iphi, *m_r, *m_ir, *m_stage, *m_istage;
};
void		make_small_avx2(short *a, int n, short q, short q_1);
void		multiply_ntt_avx2(short *dst, short const *a1, short const *a2, NTT_params_AVX2 const &p);
void		number_transform_initialize_avx2(short n, short q, short w, short sqrt_w, bool anti_cyclic, NTT_params_AVX2 &p);
void		number_transform_destroy_avx2(NTT_params_AVX2 &p);
void		apply_NTT_avx2(short *src, short *Dst, NTT_params_AVX2 const &p, bool forward_BRP=true);
void		apply_inverse_NTT_avx2(short const *src, short *Dst, NTT_params_AVX2 const &p);
struct NTT_params_SSE
{
	bool anti_cyclic;
	short n, q, w, sqrt_w, n_1, beta_q, q_1, sbar_m;
	__m128i *m_phi, *m_iphi, *m_r, *m_ir, *m_stage, *m_istage;
};
void		make_small_sse(short *a, int n, short q, short q_1);
void		multiply_ntt_sse(short *dst, short const *a1, short const *a2, NTT_params_SSE const &p);
void		number_transform_initialize_sse(short n, short q, short w, short sqrt_w, bool anti_cyclic, NTT_params_SSE &p);
void		number_transform_destroy_sse(NTT_params_SSE &p);
void		apply_NTT_sse(short *src, short *Dst, NTT_params_SSE const &p, bool forward_BRP=true);
void		apply_inverse_NTT_sse(short const *src, short *Dst, NTT_params_SSE const &p);
struct NTT_params_IA32
{
	bool anti_cyclic;
	short n, q, w, sqrt_w, n_1, beta_q, q_1,
		beta_stg, *roots, *iroots, *phi;
};
void		make_small_ia32(short *a, int n, short q, short q_1);
void		multiply_ntt_ia32(short *dst, short const *a1, short const *a2, NTT_params_IA32 const &p);
void		number_transform_initialize_ia32(short n, short q, short w, short sqrt_w, bool anti_cyclic, NTT_params_IA32 &p);
void		number_transform_destroy_ia32(NTT_params_IA32 &p);
void		apply_NTT_ia32(short *src, short *Dst, NTT_params_IA32 const &p, bool forward_BRP=true);
void		apply_inverse_NTT_ia32(short *src, short *Dst, NTT_params_IA32 const &p);
#if PROCESSOR_ARCH>=AVX2
typedef NTT_params_AVX2				NTT_params;
#define	number_transform_initialize	number_transform_initialize_avx2
#define apply_NTT					apply_NTT_avx2
#define apply_inverse_NTT			apply_inverse_NTT_avx2
#define	number_transform_destroy	number_transform_destroy_avx2
#elif PROCESSOR_ARCH>=SSE3
typedef NTT_params_SSE				NTT_params;
#define	number_transform_initialize	number_transform_initialize_sse
#define apply_NTT					apply_NTT_sse
#define apply_inverse_NTT			apply_inverse_NTT_sse
#define number_transform_destroy	number_transform_destroy_sse
#elif PROCESSOR_ARCH>=IA_32
typedef NTT_params_IA32				NTT_params;
#define number_transform_initialize	number_transform_initialize_ia32
#define apply_NTT					apply_NTT_ia32
#define apply_inverse_NTT			apply_inverse_NTT_ia32
#define number_transform_destroy	number_transform_destroy_ia32
#endif

#if PROCESSOR_ARCH>=AVX
#define		SIMD_type	__m256i
#else
#define		SIMD_type	__m128i
#endif
_LARGE_INTEGER li;
long long	freq, ticks;
const char	endl='\n';
int			log_2(unsigned n)
{
	int logn=0;

	int sh=(n>=1<<16)<<4;	logn+=sh, n>>=sh;
	sh=(n>=1<<8)<<3;		logn+=sh, n>>=sh;
	sh=(n>=1<<4)<<2;		logn+=sh, n>>=sh;
	sh=(n>=1<<2)<<1;		logn+=sh, n>>=sh;
	sh=(n>=1<<1);			logn+=sh;

	//if(n>=0x00010000)
	//	logn+=16, n>>=16;
	//if(n>=0x00000100)
	//	logn+=8, n>>=8;
	//if(n>=0x00000010)
	//	logn+=4, n>>=4;
	//if(n>=0x00000004)
	//	logn+=2, n>>=2;
	//if(n>=0x00000002)
	//	logn+=1, n>>=1;

	//for(int n2=n;n2>1;n2>>=1)
	//	++logn;

	return logn;
}
bool		extended_euclidean_algorithm(int a, int n, int &a_1)
{
	int t0=0, r0=n,
		q1=n/a, t1=1, r1=a,
		tt=t0-q1*t1, rt=r0-q1*r1;
	while(rt!=0)
	{
		t0=t1, r0=r1;
		t1=tt, r1=rt;
		q1=r0/r1;
		tt=t0-q1*t1, rt=r0-q1*r1;
	}
	if(r1==1)//gcd(a, n)=1
	{
		a_1=t1%n;
		if(a_1<0)
			a_1+=n;
		return true;
	}
	return false;
}
int			gcd(int a, int b)
{
	for(int c=a%b;c;)
		a=b, b=c, c=a%b;
	return b;
}
bool		isprime(int a)
{
	if(a<2)
		return false;
	for(int k=2, kEnd=(int)floor(sqrt((double)a));k<=kEnd;++k)
		if(!(a%k))
			return false;
	return true;
}
void		print_table_NH(const short *a, short n)
{
	for(int k=0, kEnd=n/16;k<kEnd;++k)
	{
		auto pa=a+(k<<4);
		for(int k2=0;k2<16;++k2)
			printf(" %5d", pa[k2]);
		std::cout<<endl;
	}
	auto pa=a+n-n%16;
	for(int k=0, kEnd=n%16;k<kEnd;++k)
		printf(" %5d", pa[k]);
	std::cout<<endl;
}
void		print_table(const int *a, int n)
{
	for(int k=0, kEnd=n/16;k<kEnd;++k)
	{
		auto pa=a+(k<<4);
		for(int k2=0;k2<16;++k2)
			printf(" %5d", pa[k2]);
		std::cout<<endl;
	}
	auto pa=a+n-n%16;
	for(int k=0, kEnd=n%16;k<kEnd;++k)
		printf(" %5d", pa[k]);
	std::cout<<endl;
}
void		print_histogram(const short *a, int n, int q)
{
//	std::cout<<"\nA histogram:\n";
	int h_size=10, cmd_width=80-5;
	int *histogram=new int[h_size];
	memset(histogram, 0, h_size*sizeof(int));
	for(int kv=0;kv<n;++kv)
	{
		int slot=a[kv]*h_size/q;
		++histogram[slot];
	}
	int h_max=*histogram;
	for(int kh=1;kh<h_size;++kh)
		if(h_max<histogram[kh])
			h_max=histogram[kh];
	for(int kh=0;kh<h_size;++kh)
	{
		int w=histogram[kh];
		if(h_max>cmd_width)
			w=w*cmd_width/h_max;
		printf("\n%5d\t", kh*q/h_size);
	//	std::cout<<(kh*q/h_size)<<'\t';
		for(int kx=0;kx<w;++kx)
			std::cout<<'*';
	}
	printf("\n%5d\n", q);
//	std::cout<<q<<endl;
	delete[] histogram;
}
void		print_buffer(const void *a, int size_bytes)
{
	for(int k=0;k<size_bytes;++k)
		printf("%02x", ((unsigned char*)a)[k]);
	//unsigned *ca=(unsigned*)a;
	//int size=size_bytes>>2;
	//for(int k=0;k<size;++k)
	//	printf("%08x", ca[k]);
	std::cout<<endl;
}
short		calculate_montgomery_factor(short q, short n_mon)
{
	int beta_q=0x10000%q, m_factor=1;
	for(int k=0;k<n_mon;++k)
		m_factor=m_factor*beta_q%q;
	if(n_mon<0)
	{
		int beta_1=0;
		extended_euclidean_algorithm(beta_q, q, beta_1);
		for(int k=0;k>n_mon;--k)
			m_factor=(m_factor*beta_1)%q;
	}
	return m_factor;
}
void		print_element_hex_nnl(short const *a, int n, int q, int n_mon=0)
{
	int m_factor=calculate_montgomery_factor(q, n_mon);
	for(int k=0;k<n;)
	{
		std::cout<<endl;
		for(int k2=0;(k<n)&(k2<16);++k2, ++k)
		{
			int vk=a[k]*m_factor%q;
			printf(" %5x", vk+(q&-(vk<0))-(q&-(vk>q)));
		}
	}
}
void		print_element_nnl(short const *a, int n, int q, int n_mon=0)
{
	int m_factor=calculate_montgomery_factor(q, n_mon);
	for(int k=0;k<n;)
	{
		std::cout<<endl;
		for(int k2=0;(k<n)&(k2<16);++k2, ++k)
		{
			int vk=a[k]*m_factor%q;
			printf(" %5d", vk+(q&-(vk<0))-(q&-(vk>q)));
		}
	}
}
void		print_element(short const *a, int n, int q, int n_mon=0){print_element_nnl(a, n, q, n_mon), std::cout<<endl;}
void		print_element_small(short const *a, int n, int q, int n_mon=0)
{
	int m_factor=calculate_montgomery_factor(q, n_mon);
	for(int k=0;k<n;)
	{
		std::cout<<endl;
		for(int k2=0;(k<n)&(k2<16);++k2, ++k)
		{
			int vk=a[k]*m_factor%q;
			printf(" %5d", vk+(q&-(vk<-q/2))-(q&-(vk>q/2)));
		}
	}
//	std::cout<<endl;
}
void		print_register(__m128i reg, short q, int n_mon=0)
{
	short m_factor=calculate_montgomery_factor(q, n_mon);
	for(int k=0;k<8;++k)
	{
		int vk=reg.m128i_i16[k]*m_factor%q;
		printf(" %5d", vk-(q&-(vk>q))+(q&-(vk<0)));
	}
			//-1: 2304		1		2		3		4	5		6	7		8	9	  10	11	  12	13	  14	 15
	//const int mon[]={1, 4091, 10952, 11227, 5664, 6659, 9545, 6442, 6606, 1635, 3569, 1447, 8668, 7023, 11700, 11334};
	//const int q=12289;
	//for(int k=0;k<8;++k)
	//{
	//	int vk=reg.m128i_i16[k]*mon[n_mon]%q;
	//	printf(" %5d", vk-(q&-(vk>q))+(q&-(vk<0)));
	//}
	std::cout<<endl;
}
void		print_register(const char *a, __m128i reg, short q, int n_mon=0){std::cout<<a, print_register(reg, q, n_mon);}
void		print_register(__m256i reg, short q, int n_mon=0)
{
	short m_factor=calculate_montgomery_factor(q, n_mon);
	for(int k=0;k<16;++k)
	{
		int vk=reg.m256i_i16[k]*m_factor%q;
		printf(" %5d", vk-(q&-(vk>q))+(q&-(vk<0)));
	}
		//-1: 2304		1		2		3		4	5		6	7		8	9	  10	11	  12	13	  14	 15
	//const int mon[]={1, 4091, 10952, 11227, 5664, 6659, 9545, 6442, 6606, 1635, 3569, 1447, 8668, 7023, 11700, 11334};
	//const int q=12289;
	//for(int k=0;k<16;++k)
	//{
	//	int vk=reg.m256i_i16[k]*mon[n_mon]%q;
	//	printf(" %5d", vk-(q&-(vk>q))+(q&-(vk<0)));
	//}
	std::cout<<endl;
}
void		print_register(const char *a, __m256i reg, short q, int n_mon=0){std::cout<<a, print_register(reg, q, n_mon);}
void		print_polynomial(std::vector<int> &a)
{
	std::cout<<'(';
	int n=a.size();
	for(int k=n-1;k>=2;--k)
		std::cout<<a[k]<<'x'<<k<<" + ";
	if(n>1)
		std::cout<<a[1]<<"x + ";
	if(n)
		std::cout<<a[0];
//	std::cout<<a[n-1]<<'x'<<n-1;
//	for(int k=n-2;k>=0;--k)
//		std::cout<<" + "<<a[k]<<'x'<<k;
	std::cout<<')';
}
void		print_matrix(int const *A, int h, int w, int log10q)
{
	for(int k=0;k<h;++k)
	{
		for(int k2=0;k2<w;++k2)
			printf("%*d", log10q+1, A[k*w+k2]);
		std::cout<<'\n';
	}
}

class	AES
{
	static unsigned char (*key)[4][4], Dkey[11][4][4];
	static unsigned char mult_gf_2_8(unsigned char a, unsigned char b);
	static unsigned char mult_by_4_gf_2_8(unsigned char x){return mult_by_2[mult_by_2[x]];}
	static unsigned char mult_by_8_gf_2_8(unsigned char x){return mult_by_2[mult_by_2[mult_by_2[x]]];}
	static int leftmost_up_bit_pos(int x);
	static int mult_gf_2(int a, int b);
	static int divide_gf_2(int a, int b, int *r);
	static bool mult_inv_gf_2_8(unsigned char x, unsigned char &x_1);
	static void s_box_step_4(unsigned char &x);
	static void s_box_1_step_3(unsigned char &x);
	static void add_round_key(int round);
	static void substitute_bytes();
	static void inverse_sub_bytes();
	static void shift_rows();
	static void inverse_shift_rows();
	static void mix_columns();
	static void inverse_mix_columns();
public:
	static unsigned char s_box[256], s_box_1[256], mult_by_2[256], x_pow_i_4_1[11];
	static unsigned int DK0[256], DK1[256], DK2[256], DK3[256],
		E0[256], E1[256], E2[256], E3[256], SBS8[256], SBS16[256], SBS24[256],
		D0[256], D1[256], D2[256], D3[256], SB1S8[256], SB1S16[256], SB1S24[256];
	static void initiate();
	static void expand_key(unsigned char *key);
	static void encrypt(unsigned char *text);
	static void encrypt(unsigned char *text, unsigned char *key);
	static void decrypt(unsigned char *text);
	static void decrypt(unsigned char *text, unsigned char *key);
};
unsigned char	AES::s_box[256], AES::s_box_1[256], AES::mult_by_2[256], AES::x_pow_i_4_1[11];
unsigned int	AES::DK0[256], AES::DK1[256], AES::DK2[256], AES::DK3[256],
	AES::E0[256], AES::E1[256], AES::E2[256], AES::E3[256], AES::SBS8[256], AES::SBS16[256], AES::SBS24[256],
	AES::D0[256], AES::D1[256], AES::D2[256], AES::D3[256], AES::SB1S8[256], AES::SB1S16[256], AES::SB1S24[256];
unsigned char	(*AES::key)[4][4], AES::Dkey[11][4][4];
unsigned char	AES::mult_gf_2_8(unsigned char a, unsigned char b)
{
	int result=0;
	for(int k=0;k<8;++k)
	{
		if(b&1<<k)
			result^=a;
		a=a&0x80?a<<1^0x1B:a<<1;
	}
	return result;
}
int			AES::leftmost_up_bit_pos(int x)
{
	int k=31;
	for(;k>=0;--k)
		if(x&1<<k)
			break;
	return k;
}
int			AES::mult_gf_2(int a, int b)
{
	int result=0;
	for(int k=0;k<32;++k)
		if(b&1<<k)
			result^=a<<k;
	return result;
}
int			AES::divide_gf_2(int a, int b, int *r=0)
{
	int q=0;
	for(int xb=leftmost_up_bit_pos(b);a>=b;)
	{
		int xa_b=leftmost_up_bit_pos(a)-xb;
		q^=1<<xa_b, a^=b<<xa_b;
	}
	if(r)
		*r=a;
	return q;
}
bool		AES::mult_inv_gf_2_8(unsigned char x, unsigned char &x_1)
{
	int Q, A[3]={1, 0, 0x11B}, B[3]={0, 1, x}, T[3];
	for(;B[2]!=1&&B[2]!=0;)
	{
		Q=divide_gf_2(A[2], B[2]);
		T[0]=A[0], T[1]=A[1], T[2]=A[2];
		A[0]=B[0], A[1]=B[1], A[2]=B[2];
		B[0]=T[0]^mult_gf_2(Q, B[0]), B[1]=T[1]^mult_gf_2(Q, B[1]), B[2]=T[2]^mult_gf_2(Q, B[2]);
	}
	if(B[2])
	{
		x_1=B[1];
		return true;
	}
	return false;
}
void		AES::s_box_step_4(unsigned char &x)
{
	int result=0, c=0x63;
	for(int k=0;k<8;++k)
		result^=(x>>k&1^x>>(k+4)%8&1^x>>(k+5)%8&1^x>>(k+6)%8&1^x>>(k+7)%8&1^c>>k&1)<<k;
	x=result;
}
void		AES::s_box_1_step_3(unsigned char &x)
{
	int result=0, d=0x05;
	for(int k=0;k<8;++k)
		result^=(x>>(k+2)%8&1^x>>(k+5)%8&1^x>>(k+7)%8&1^d>>k&1)<<k;
	x=result;
}
void		AES::initiate()
{
	for(int k=0;k<256;++k)
	{
		s_box[k]=k;
		mult_inv_gf_2_8(s_box[k], s_box[k]);
		s_box_step_4(s_box[k]);

		s_box_1[k]=k;
		s_box_1_step_3(s_box_1[k]);
		mult_inv_gf_2_8(s_box_1[k], s_box_1[k]);
	/*	printf("%02X ", (unsigned char)s_box_1[k]);
		if(!((k+1)%16))
			printf("\n");*/

		mult_by_2[k]=mult_gf_2_8(k, 2);
	}
	for(int k=0;k<256;++k)//2113	3211	1321	1132
	{
		unsigned char c=s_box[k], c2=mult_by_2[c], c3=c2^c;
		E0[k]=c2|c<<8|c<<16|c3<<24, E1[k]=c3|c2<<8|c<<16|c<<24, E2[k]=c|c3<<8|c2<<16|c<<24, E3[k]=c|c<<8|c3<<16|c2<<24, SBS8[k]=c<<8, SBS16[k]=c<<16, SBS24[k]=c<<24;
	}
	for(int k=0;k<256;++k)//E9DB	BE9D	DBE9	9DBE
	{
		unsigned char c=k, c2=mult_by_2[c], c3=c2^c, c4=mult_by_2[c2], c8=mult_by_2[c4], c9=c8^c, cB=c8^c3, cC=c8^c4, cD=cC^c, cE=cC^c2;
		DK0[k]=cE|c9<<8|cD<<16|cB<<24, DK1[k]=cB|cE<<8|c9<<16|cD<<24, DK2[k]=cD|cB<<8|cE<<16|c9<<24, DK3[k]=c9|cD<<8|cB<<16|cE<<24;
	}
	for(int k=0;k<256;++k)//E9DB	BE9D	DBE9	9DBE
	{
		unsigned char c=s_box_1[k], c2=mult_by_2[c], c3=c2^c, c4=mult_by_2[c2], c8=mult_by_2[c4], c9=c8^c, cB=c8^c3, cC=c8^c4, cD=cC^c, cE=cC^c2;
		D0[k]=cE|c9<<8|cD<<16|cB<<24, D1[k]=cB|cE<<8|c9<<16|cD<<24, D2[k]=cD|cB<<8|cE<<16|c9<<24, D3[k]=c9|cD<<8|cB<<16|cE<<24, SB1S8[k]=c<<8, SB1S16[k]=c<<16, SB1S24[k]=c<<24;
	}
	{
		unsigned char smiley=0x8D;
		for(int k=0;k<11;++k)
			smiley=mult_by_2[x_pow_i_4_1[k]=smiley];
	}
}
void		AES::expand_key(unsigned char key[11*16])
{
	*(unsigned char**)&AES::key=key;
	for(int k=16;k<176;k+=4)
	{
		if(!(k%16))
			key[k  ]=s_box[key[k-3]]^x_pow_i_4_1[k/16]^key[k-16],
			key[k+1]=s_box[key[k-2]]^key[k-15],
			key[k+2]=s_box[key[k-1]]^key[k-14],
			key[k+3]=s_box[key[k-4]]^key[k-13];
		else
			key[k  ]=key[k-4]^key[k-16],
			key[k+1]=key[k-3]^key[k-15],
			key[k+2]=key[k-2]^key[k-14],
			key[k+3]=key[k-1]^key[k-13];
	}
	for(int k=0;k<16;k+=4)
		*(int*)((char*)Dkey+k)=*(int*)(key+160+k);
	for(int k=16;k<160;k+=16)
		for(int k2=0;k2<16;k2+=4)
			*(int*)((char*)Dkey+k+k2)=DK0[key[160-k+k2]]^DK1[key[160-k+k2+1]]^DK2[key[160-k+k2+2]]^DK3[key[160-k+k2+3]];
	for(int k=0;k<16;k+=4)
		*(int*)((char*)Dkey+160+k)=*(int*)(key+k);

/*	for(int k=0;k<16;k+=4)
		*(int*)((char*)Dkey+k)=*(int*)(key+k);
	for(int k=16;k<160;k+=4)
		*(int*)((char*)Dkey+k)=DK0[key[k]]^DK1[key[k+1]]^DK2[key[k+2]]^DK3[key[k+3]];
	for(int k=160;k<176;k+=4)
		*(int*)((char*)Dkey+k)=*(int*)(key+k);*/
}
void		AES::encrypt(unsigned char *text, unsigned char *key){expand_key(key), encrypt(text);}
void		AES::encrypt(unsigned char text[16])//http://software.intel.com/en-us/articles/optimizing-performance-of-the-aes-algorithm-for-the-intel-pentiumr-4-processor/
{
	//unsigned char text2[16];
	//for(int k=0;k<16;++k)
	//	text2[k]=text[k];
#ifdef PROCESSOR_ARCH_AES_NI
	__m128i m_text=_mm_loadu_si128((__m128i*)text);
	__m128i m_key=_mm_loadu_si128((__m128i*)key);
	//std::cout<<"key:\t", print_buffer(key, 16);//
	//std::cout<<"text:\t", print_buffer(text, 16);//
	m_text=_mm_xor_si128(m_text, m_key);
	//print_buffer(&m_text, 16);//
	m_key=_mm_loadu_si128((__m128i*)key+1);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+2);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+3);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+4);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+5);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+6);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+7);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+8);
	m_text=_mm_aesenc_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)key+9);
	m_text=_mm_aesenc_si128(m_text, m_key);
	//for(int r=1;r<10;++r)
	//{
	//	m_key=_mm_loadu_si128((__m128i*)key+r);
	//	m_text=_mm_aesenc_si128(m_text, m_key);
	//	print_buffer(&m_text, 16);//
	//}
	m_key=_mm_loadu_si128((__m128i*)key+10);
	m_text=_mm_aesenclast_si128(m_text, m_key);
	//print_buffer(&m_text, 16);//
	_mm_storeu_si128((__m128i*)text, m_text);
#else//*/
	unsigned char temp0[4][4], temp1[4][4];

//	*(long long*)temp0[0]=*(long long*) text   ^*(long long*)key[0][0];
//	*(long long*)temp0[2]=*(long long*)(text+8)^*(long long*)key[0][2];

	*(int*)temp0[0]=*(int*)key[0][0]^*(int*) text    ;
	*(int*)temp0[1]=*(int*)key[0][1]^*(int*)(text+ 4);
	*(int*)temp0[2]=*(int*)key[0][2]^*(int*)(text+ 8);
	*(int*)temp0[3]=*(int*)key[0][3]^*(int*)(text+12);
	//std::cout<<"key:\t", print_buffer(key, 16);//
	//std::cout<<"text:\t", print_buffer(text, 16);//
	//print_buffer(temp0, 16);//
	for(int r=1;r<8;r+=2)
	{
		*(int*)temp1[0]=*(int*)key[r  ][0]^E0[temp0[0][0]]^E1[temp0[1][1]]^E2[temp0[2][2]]^E3[temp0[3][3]];
		*(int*)temp1[1]=*(int*)key[r  ][1]^E0[temp0[1][0]]^E1[temp0[2][1]]^E2[temp0[3][2]]^E3[temp0[0][3]];
		*(int*)temp1[2]=*(int*)key[r  ][2]^E0[temp0[2][0]]^E1[temp0[3][1]]^E2[temp0[0][2]]^E3[temp0[1][3]];
		*(int*)temp1[3]=*(int*)key[r  ][3]^E0[temp0[3][0]]^E1[temp0[0][1]]^E2[temp0[1][2]]^E3[temp0[2][3]];
		*(int*)temp0[0]=*(int*)key[r+1][0]^E0[temp1[0][0]]^E1[temp1[1][1]]^E2[temp1[2][2]]^E3[temp1[3][3]];
		*(int*)temp0[1]=*(int*)key[r+1][1]^E0[temp1[1][0]]^E1[temp1[2][1]]^E2[temp1[3][2]]^E3[temp1[0][3]];
		*(int*)temp0[2]=*(int*)key[r+1][2]^E0[temp1[2][0]]^E1[temp1[3][1]]^E2[temp1[0][2]]^E3[temp1[1][3]];
		*(int*)temp0[3]=*(int*)key[r+1][3]^E0[temp1[3][0]]^E1[temp1[0][1]]^E2[temp1[1][2]]^E3[temp1[2][3]];
		//print_buffer(temp1, 16);//
		//print_buffer(temp0, 16);//
	}
	*(int*)temp1[0]=*(int*)key[9][0]^E0[temp0[0][0]]^E1[temp0[1][1]]^E2[temp0[2][2]]^E3[temp0[3][3]];
	*(int*)temp1[1]=*(int*)key[9][1]^E0[temp0[1][0]]^E1[temp0[2][1]]^E2[temp0[3][2]]^E3[temp0[0][3]];
	*(int*)temp1[2]=*(int*)key[9][2]^E0[temp0[2][0]]^E1[temp0[3][1]]^E2[temp0[0][2]]^E3[temp0[1][3]];
	*(int*)temp1[3]=*(int*)key[9][3]^E0[temp0[3][0]]^E1[temp0[0][1]]^E2[temp0[1][2]]^E3[temp0[2][3]];
	*(int*) text    =*(int*)key[10][0]^s_box[temp1[0][0]]^SBS8[temp1[1][1]]^SBS16[temp1[2][2]]^SBS24[temp1[3][3]];
	*(int*)(text+ 4)=*(int*)key[10][1]^s_box[temp1[1][0]]^SBS8[temp1[2][1]]^SBS16[temp1[3][2]]^SBS24[temp1[0][3]];
	*(int*)(text+ 8)=*(int*)key[10][2]^s_box[temp1[2][0]]^SBS8[temp1[3][1]]^SBS16[temp1[0][2]]^SBS24[temp1[1][3]];
	*(int*)(text+12)=*(int*)key[10][3]^s_box[temp1[3][0]]^SBS8[temp1[0][1]]^SBS16[temp1[1][2]]^SBS24[temp1[2][3]];
	//print_buffer(temp1, 16);//
	//std::cout<<"ct:\t", print_buffer(text, 16), std::cout<<endl;//
#endif
	//for(int k=0;k<16;++k)text2[k]=((char*)temp0)[k];

/*	add_round_key(0);//round 0
	for(int k=1;k<10;++k)//rounds 1~9
	{
		substitute_bytes();
		shift_rows();
		mix_columns();
		add_round_key(k);
	}
	substitute_bytes();//round 10
	shift_rows();
	add_round_key(10);*/
}
void		AES::decrypt(unsigned char *text, unsigned char *key){expand_key(key), decrypt(text);}
void		AES::decrypt(unsigned char *text)
{
#ifdef PROCESSOR_ARCH_AES_NI
	__m128i m_text=_mm_loadu_si128((__m128i*)text);
	__m128i m_key=_mm_loadu_si128((__m128i*)Dkey);
	m_text=_mm_xor_si128(m_text, m_key);
	
	m_key=_mm_loadu_si128((__m128i*)Dkey+1);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+2);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+3);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+4);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+5);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+6);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+7);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+8);
	m_text=_mm_aesdec_si128(m_text, m_key);
	m_key=_mm_loadu_si128((__m128i*)Dkey+9);
	m_text=_mm_aesdec_si128(m_text, m_key);
	//for(int r=1;r<10;++r)
	//{
	//	m_key=_mm_loadu_si128((__m128i*)Dkey+r);
	//	m_text=_mm_aesdec_si128(m_text, m_key);
	//}
	m_key=_mm_loadu_si128((__m128i*)Dkey+10);
	m_text=_mm_aesdeclast_si128(m_text, m_key);
	_mm_storeu_si128((__m128i*)text, m_text);
#else//*/
	unsigned char temp0[4][4], temp1[4][4];

//	*(long long*)temp[0]=*(long long*)text^*(long long*)key[round][0];
//	*(long long*)temp[2]=*(long long*)(text+8)^*(long long*)key[round][2];

	*(int*)temp0[0]=*(int*) text    ^*(int*)Dkey[0][0];
	*(int*)temp0[1]=*(int*)(text+ 4)^*(int*)Dkey[0][1];
	*(int*)temp0[2]=*(int*)(text+ 8)^*(int*)Dkey[0][2];
	*(int*)temp0[3]=*(int*)(text+12)^*(int*)Dkey[0][3];
	for(int r=1;r<8;r+=2)
	{
		*(int*)temp1[0]=*(int*)Dkey[r  ][0]^D0[temp0[0][0]]^D1[temp0[3][1]]^D2[temp0[2][2]]^D3[temp0[1][3]];
		*(int*)temp1[1]=*(int*)Dkey[r  ][1]^D0[temp0[1][0]]^D1[temp0[0][1]]^D2[temp0[3][2]]^D3[temp0[2][3]];
		*(int*)temp1[2]=*(int*)Dkey[r  ][2]^D0[temp0[2][0]]^D1[temp0[1][1]]^D2[temp0[0][2]]^D3[temp0[3][3]];
		*(int*)temp1[3]=*(int*)Dkey[r  ][3]^D0[temp0[3][0]]^D1[temp0[2][1]]^D2[temp0[1][2]]^D3[temp0[0][3]];
		*(int*)temp0[0]=*(int*)Dkey[r+1][0]^D0[temp1[0][0]]^D1[temp1[3][1]]^D2[temp1[2][2]]^D3[temp1[1][3]];
		*(int*)temp0[1]=*(int*)Dkey[r+1][1]^D0[temp1[1][0]]^D1[temp1[0][1]]^D2[temp1[3][2]]^D3[temp1[2][3]];
		*(int*)temp0[2]=*(int*)Dkey[r+1][2]^D0[temp1[2][0]]^D1[temp1[1][1]]^D2[temp1[0][2]]^D3[temp1[3][3]];
		*(int*)temp0[3]=*(int*)Dkey[r+1][3]^D0[temp1[3][0]]^D1[temp1[2][1]]^D2[temp1[1][2]]^D3[temp1[0][3]];
	}
	*(int*)temp1[0]=*(int*)Dkey[9][0]^D0[temp0[0][0]]^D1[temp0[3][1]]^D2[temp0[2][2]]^D3[temp0[1][3]];
	*(int*)temp1[1]=*(int*)Dkey[9][1]^D0[temp0[1][0]]^D1[temp0[0][1]]^D2[temp0[3][2]]^D3[temp0[2][3]];
	*(int*)temp1[2]=*(int*)Dkey[9][2]^D0[temp0[2][0]]^D1[temp0[1][1]]^D2[temp0[0][2]]^D3[temp0[3][3]];
	*(int*)temp1[3]=*(int*)Dkey[9][3]^D0[temp0[3][0]]^D1[temp0[2][1]]^D2[temp0[1][2]]^D3[temp0[0][3]];
	*(int*) text    =*(int*)Dkey[10][0]^s_box_1[temp1[0][0]]^SB1S8[temp1[3][1]]^SB1S16[temp1[2][2]]^SB1S24[temp1[1][3]];
	*(int*)(text+ 4)=*(int*)Dkey[10][1]^s_box_1[temp1[1][0]]^SB1S8[temp1[0][1]]^SB1S16[temp1[3][2]]^SB1S24[temp1[2][3]];
	*(int*)(text+ 8)=*(int*)Dkey[10][2]^s_box_1[temp1[2][0]]^SB1S8[temp1[1][1]]^SB1S16[temp1[0][2]]^SB1S24[temp1[3][3]];
	*(int*)(text+12)=*(int*)Dkey[10][3]^s_box_1[temp1[3][0]]^SB1S8[temp1[2][1]]^SB1S16[temp1[1][2]]^SB1S24[temp1[0][3]];
#endif

/*	*(int*)temp0[0]=*(int*) text    ^*(int*)Dkey[10][0];
	*(int*)temp0[1]=*(int*)(text+ 4)^*(int*)Dkey[10][1];
	*(int*)temp0[2]=*(int*)(text+ 8)^*(int*)Dkey[10][2];
	*(int*)temp0[3]=*(int*)(text+12)^*(int*)Dkey[10][3];
	for(int r=9;r>2;r-=2)
	{
		*(int*)temp1[0]=*(int*)Dkey[r  ][0]^D0[temp0[0][0]]^D1[temp0[3][1]]^D2[temp0[2][2]]^D3[temp0[1][3]];
		*(int*)temp1[1]=*(int*)Dkey[r  ][1]^D0[temp0[1][0]]^D1[temp0[0][1]]^D2[temp0[3][2]]^D3[temp0[2][3]];
		*(int*)temp1[2]=*(int*)Dkey[r  ][2]^D0[temp0[2][0]]^D1[temp0[1][1]]^D2[temp0[0][2]]^D3[temp0[3][3]];
		*(int*)temp1[3]=*(int*)Dkey[r  ][3]^D0[temp0[3][0]]^D1[temp0[2][1]]^D2[temp0[1][2]]^D3[temp0[0][3]];
		*(int*)temp0[0]=*(int*)Dkey[r-1][0]^D0[temp1[0][0]]^D1[temp1[3][1]]^D2[temp1[2][2]]^D3[temp1[1][3]];
		*(int*)temp0[1]=*(int*)Dkey[r-1][1]^D0[temp1[1][0]]^D1[temp1[0][1]]^D2[temp1[3][2]]^D3[temp1[2][3]];
		*(int*)temp0[2]=*(int*)Dkey[r-1][2]^D0[temp1[2][0]]^D1[temp1[1][1]]^D2[temp1[0][2]]^D3[temp1[3][3]];
		*(int*)temp0[3]=*(int*)Dkey[r-1][3]^D0[temp1[3][0]]^D1[temp1[2][1]]^D2[temp1[1][2]]^D3[temp1[0][3]];
	}
	*(int*)temp1[0]=*(int*)Dkey[1][0]^D0[temp0[0][0]]^D1[temp0[3][1]]^D2[temp0[2][2]]^D3[temp0[1][3]];
	*(int*)temp1[1]=*(int*)Dkey[1][1]^D0[temp0[1][0]]^D1[temp0[0][1]]^D2[temp0[3][2]]^D3[temp0[2][3]];
	*(int*)temp1[2]=*(int*)Dkey[1][2]^D0[temp0[2][0]]^D1[temp0[1][1]]^D2[temp0[0][2]]^D3[temp0[3][3]];
	*(int*)temp1[3]=*(int*)Dkey[1][3]^D0[temp0[3][0]]^D1[temp0[2][1]]^D2[temp0[1][2]]^D3[temp0[0][3]];
	*(int*) text    =*(int*)Dkey[0][0]^s_box_1[temp1[0][0]]^SB1S8[temp1[3][1]]^SB1S16[temp1[2][2]]^SB1S24[temp1[1][3]];
	*(int*)(text+ 4)=*(int*)Dkey[0][1]^s_box_1[temp1[1][0]]^SB1S8[temp1[0][1]]^SB1S16[temp1[3][2]]^SB1S24[temp1[2][3]];
	*(int*)(text+ 8)=*(int*)Dkey[0][2]^s_box_1[temp1[2][0]]^SB1S8[temp1[1][1]]^SB1S16[temp1[0][2]]^SB1S24[temp1[3][3]];
	*(int*)(text+12)=*(int*)Dkey[0][3]^s_box_1[temp1[3][0]]^SB1S8[temp1[2][1]]^SB1S16[temp1[1][2]]^SB1S24[temp1[0][3]];*/

/*	add_round_key(10);//round 0
	for(int k=9;k>0;--k)//rounds 1~9
	{
		inverse_shift_rows();
		inverse_sub_bytes();
		add_round_key(k);
		inverse_mix_columns();
	}
	inverse_shift_rows();//round 10
	inverse_sub_bytes();
	add_round_key(0);*/
}

//DRBG-128
const unsigned	drbg_size=16;
long long		drbg_ctr[2]={0}, drbg_str[2]={0};
unsigned char	drbg_key[11*drbg_size]={0}, drbg_pos=0;
void		drbg_generate()
{
	memcpy(drbg_str, drbg_ctr, drbg_size);
//	std::cout<<"drbg ctr:\t", printf("%016llx%016llx\n", drbg_ctr[1], drbg_ctr[0]);//
	AES::encrypt((unsigned char*)drbg_str);
//	std::cout<<"drbg str:\t", print_buffer(message, block_size);
			
	int flag=~drbg_ctr[0]==0;
	++drbg_ctr[0], drbg_ctr[0]&=(long long)-!flag, drbg_ctr[1]+=flag;
	drbg_pos=0;
}
void		drbg_start(const unsigned char seed[16])
{
	memcpy(drbg_key, seed, drbg_size);
	memset(drbg_key+drbg_size, 0, 10*drbg_size);
	AES::expand_key(drbg_key);
	memset(drbg_ctr, 0, drbg_size);
	drbg_generate();
}
void		drbg_get(void *a, unsigned size_in_bytes)
{
	unsigned char *str=(unsigned char*)drbg_str, *aa=(unsigned char*)a;
	if(drbg_pos+size_in_bytes<drbg_size)
	{
		memcpy(a, str+drbg_pos, size_in_bytes);
		drbg_pos+=size_in_bytes;
	}
	else
	{
		unsigned a_pos=drbg_size-drbg_pos;
		memcpy(a, str+drbg_pos, a_pos);
		drbg_generate();
		for(;a_pos+drbg_size<=size_in_bytes;)
		{
			memcpy(aa+a_pos, drbg_str, drbg_size);
			a_pos+=drbg_size;
			drbg_generate();
		}
		if(a_pos<size_in_bytes)
		{
			drbg_pos=size_in_bytes-a_pos;
			memcpy(aa+a_pos, drbg_str, drbg_pos);
			if(drbg_pos>=drbg_size)
				drbg_generate();
		}
	}
}
void		drbg_squeeze(const unsigned char *in, unsigned in_size)//in_size multiple of 16, collision resistance of 2^64
{
	memset(drbg_str, 0, drbg_size);
	for(unsigned i=0;i<in_size;i+=drbg_size)
	{
		memcpy(drbg_key, in+i, drbg_size);
		memset(drbg_key+drbg_size, 0, 10*drbg_size);
		AES::expand_key(drbg_key);
		AES::encrypt((unsigned char*)drbg_str);
	}
	drbg_start((unsigned char*)drbg_str);
}
void		drbg_shake128(const unsigned char *in, unsigned in_size, unsigned char *out, unsigned out_size)
{
	if(in_size&0xF)
	{
		unsigned msg_size=(in_size+15)&0xFFFFFFF0;
		unsigned char *msg=(unsigned char*)malloc(msg_size);
		memcpy(msg, in, in_size);
		memset(msg+in_size, 0, 16-(in_size&0xF));
		drbg_squeeze(msg, msg_size);
		free(msg);
	}
	else
		drbg_squeeze(in, in_size);
	drbg_get(out, out_size);
}

//SHA-3
//http://keccak.noekeon.org/tweetfips202.html
static inline unsigned long long load64(const unsigned char *x)
{
	unsigned long long u=*(unsigned long long*)x;
	return u;
//	unsigned i;
//	unsigned long long u=0;
//	for(i=0; i<8; ++i)
//		u<<=8, u|=x[7-i];
//	return u;
}
static inline void store64(unsigned char *x, unsigned long long u)
{
	*(unsigned long long*)x=u;
//	unsigned i;
//	for(i=0; i<8; ++i)
//		x[i]=u, u>>=8;
}
static inline void xor64(unsigned char *x, unsigned long long u)
{
//	*(__m64*)x=_mm_xor_si64(*(__m64*)x, (__m64&)u);
	*(unsigned long long*)x^=u;
//	unsigned i;
//	for(i=0; i<8; ++i)
//		x[i]^=u, u>>=8;
//	_mm_empty();
}
int			LFSR86540(unsigned char *R)
{
	int result=*R&1;
	*R=(*R<<1)^(*R&0x80?0x71:0);//39230
//	*R=*R<<1^(0x71&-((*R&0x80)!=0));//39650
	return result;
//	return (*R&2)>>1;
}
void		KeccakF1600(void *s)
{
    unsigned r, x, y, i, j, Y;
	unsigned char R=0x01;
	unsigned long long C[5], D;
    for(i=0;i<24;++i)
	{
		for(x=0;x<5;++x)	//theta
		{
			C[x]=(long long&)_mm_xor_si64(((__m64*)s)[x], ((__m64*)s)[x+5]);//39650
			C[x]=(long long&)_mm_xor_si64((__m64&)C[x], ((__m64*)s)[x+10]);
			C[x]=(long long&)_mm_xor_si64((__m64&)C[x], ((__m64*)s)[x+15]);
			C[x]=(long long&)_mm_xor_si64((__m64&)C[x], ((__m64*)s)[x+20]);
		}
		//	C[x]=((unsigned long long*)s)[x]^((unsigned long long*)s)[x+5]^((unsigned long long*)s)[x+10]^((unsigned long long*)s)[x+15]^((unsigned long long*)s)[x+20];//41110
			//C[x]=load64((unsigned char*)s+8*(x+5*0))//40130
			//	^load64((unsigned char*)s+8*(x+5*1))
			//	^load64((unsigned char*)s+8*(x+5*2))
			//	^load64((unsigned char*)s+8*(x+5*3))
			//	^load64((unsigned char*)s+8*(x+5*4));
		for(x=0;x<5;++x)
		{
			(__m64&)D=_mm_xor_si64((__m64&)C[(x+4)%5], _mm_xor_si64(_mm_slli_si64((__m64&)C[(x+1)%5], 1), _mm_srli_si64((__m64&)C[(x+1)%5], 64-1)));//38200
		//	D=C[(x+4)%5]^((((unsigned long long)C[(x+1)%5])<<1)^(((unsigned long long)C[(x+1)%5])>>(64-1)));
			for(y=0;y<5;++y)
				((__m64*)s)[x+5*y]=_mm_xor_si64(((__m64*)s)[x+5*y], (__m64&)D);//39250
			//	xor64((unsigned char*)s+8*(x+5*y),D);
		}
		x=1, y=r=0;			//rho*pi
		D=load64((unsigned char*)s+8*(x+5*y));
		for(j=0;j<24;++j)
		{
			r+=j+1;
			Y=(2*x+3*y)%5, x=y, y=Y;
			C[0]=load64((unsigned char*)s+8*(x+5*y));
			store64((unsigned char*)s+8*(x+5*y), ((((unsigned long long)D)<<r%64)^(((unsigned long long)D)>>(64-r%64))));//37250	38200
		//	((__m64*)s)[x+5*y]=_mm_xor_si64(_mm_sll_si64((__m64&)D, _mm_set_pi32(0, r&0x3F)), _mm_srl_si64((__m64&)D, _mm_set_pi32(0, 64-(r&0x3F))));//38760	38580	38380
			D=C[0];
		}
		for(y=0;y<5;++y)	//chi
		{
			for(x=0;x<5;++x)
			//	C[x]=((unsigned long long*)s)[x+5*y];//39490
				C[x]=load64((unsigned char*)s+8*(x+5*y));//39120
			for(x=0;x<5;++x)
				store64((unsigned char*)s+8*(x+5*y), C[x]^((~C[(x+1)%5])&C[(x+2)%5]));//37970
			//	((__m64*)s)[x+5*y]=_mm_xor_si64((__m64&)C[x], _mm_and_si64(_mm_xor_si64((__m64&)C[(x+1)%5], _mm_set1_pi32(-1)), (__m64&)C[(x+2)%5]));//49920
		}
		for(j=0;j<7;++j)	//iota
			if(LFSR86540(&R))
				xor64((unsigned char*)s+8*(0 +5*0), (unsigned long long)1<<((1<<j)-1));//39350
			//{
			//	unsigned long long temp=1ULL<<((1<<j)-1);
			//	*((__m64*)s)=_mm_xor_si64(*((__m64*)s), (__m64&)temp);//42390
			//}
			//	*((__m64*)s)=_mm_xor_si64(*((__m64*)s), _mm_set_pi32(0, 1<<((1<<j)-1)));//X 40920
    }
	_mm_empty();
}
void		Keccak(unsigned r, unsigned c, const unsigned char *in, unsigned long long inLen, unsigned char sfx, unsigned char *out, unsigned long long outLen)
{
	__declspec(align(8)) unsigned char s[200]={0};//initialize
	unsigned R=r/8, i, b=0;
	memset(s, 0, 200);
//	for(i=0; i<200; ++i)
//		s[i]=0;
	while(inLen>0)		//absorb
	{
		b=inLen<R?unsigned(inLen):R;
		for(i=0;i<b;++i)
			s[i]^=in[i];
		in+=b, inLen-=b;
		if(b==R)
		{
			KeccakF1600(s);
			b=0;
		}
	}
	s[b]^=sfx;			//pad
	if((sfx&0x80)&&(b==(R-1)))
		KeccakF1600(s);
	s[R-1]^=0x80;
	KeccakF1600(s);
	while(outLen>0)		//squeeze
	{
		b=outLen<R?unsigned(outLen):R;
		for(i=0;i<b;++i)
			out[i]=s[i];
		out+=b, outLen-=b;
		if(outLen>0)
			KeccakF1600(s);
	}
}
#ifdef XOF_USE_KECCAK_TINY
void		FIPS202_SHAKE128(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){shake128(out, (unsigned)outLen, in, (unsigned)inLen);}
void		FIPS202_SHAKE256(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){shake256(out, (unsigned)outLen, in, (unsigned)inLen);}
#elif defined XOF_USE_TWEETFIPS202
void		FIPS202_SHAKE128(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){Keccak(1344, 256, in, inLen, 0x1F, out, outLen);}
void		FIPS202_SHAKE256(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){Keccak(1088, 512, in, inLen, 0x1F, out, outLen);}
#elif defined XOF_USE_DRBG_AES128
void		FIPS202_SHAKE128(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){drbg_shake128(in, (unsigned)inLen, out, (unsigned)outLen);}
void		FIPS202_SHAKE256(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){drbg_shake128(in, (unsigned)inLen, out, (unsigned)outLen);}
#else
void		FIPS202_SHAKE128(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){memset(out, 0, (unsigned)outLen);}
void		FIPS202_SHAKE256(const unsigned char *in, unsigned long long inLen, unsigned char *out, unsigned long long outLen){memset(out, 0, (unsigned)outLen);}
#endif
void		FIPS202_SHA3_224(const unsigned char *in, unsigned long long inLen, unsigned char *out){Keccak(1152,  448, in, inLen, 0x06, out, 28);}
void		FIPS202_SHA3_256(const unsigned char *in, unsigned long long inLen, unsigned char *out){Keccak(1088,  512, in, inLen, 0x06, out, 32);}
void		FIPS202_SHA3_384(const unsigned char *in, unsigned long long inLen, unsigned char *out){Keccak( 832,  768, in, inLen, 0x06, out, 48);}
void		FIPS202_SHA3_512(const unsigned char *in, unsigned long long inLen, unsigned char *out){Keccak( 576, 1024, in, inLen, 0x06, out, 64);}

bool		use_rand=true;
HCRYPTPROV	hProv=0;
inline short montgomery_reduction(int x, short q, short q_1, short beta2)
{
	short v=short(*(short*)&x*q_1);
	v=v*q>>16, v=((short*)&x)[1]-v, v+=q&-(v<0);
	x=v*beta2;
	v=short(*(short*)&x*q_1), v=v*q>>16, v=((short*)&x)[1]-v, v+=q&-(v<0);
	return v;
}
inline short kyber_montgomery_reduction(int x)
{
	const short q=7681, beta_q=4088, q_1=-7679, beta2=5569;
	short v=short(*(short*)&x*q_1);
	v=v*q>>16, v=((short*)&x)[1]-v, v+=q&-(v<0);
	x=v*beta2;
	v=short(*(short*)&x*q_1), v=v*q>>16, v=((short*)&x)[1]-v, v+=q&-(v<0);
	return v;
}
short*		pack_bits(const unsigned short *a1, int a1_size, unsigned char coeff_size, unsigned char logbeta, int align)
{
	int a2_size=a1_size*coeff_size/logbeta;
//	int u_size=vector_size*11/(8*sizeof(short));
	unsigned short *a2=(unsigned short*)_aligned_malloc(a2_size*sizeof(unsigned short), align);
//	short mask=(1<<coeff_size)-1;
	for(int k=0, k2=0, start=0, ammount=0;k2<a2_size;)
	{
		a2[k2]=a1[k]>>-start, ++k;
		for(ammount=start+coeff_size;ammount<logbeta;ammount+=coeff_size, ++k)
			a2[k2]|=a1[k]<<ammount;
		k-=ammount>logbeta;
		++k2, start=ammount-logbeta-coeff_size, start+=coeff_size&-(start<=-coeff_size);
	}
//	_aligned_free(a1);
	return (short*)a2;
}
short*		unpack_bits(const unsigned short *a1, int a1_size, unsigned char coeff_size, unsigned char logbeta, int align)
{
/*	int a2_size=a1_size*coeff_size/logbeta;
//	int a2_size=a1_size*11/(8*sizeof(short));
	unsigned short *a2=(unsigned short*)_aligned_malloc(a2_size*sizeof(unsigned short), align);
	short mask=0x07FF;
	for(int k=0, k2=0, start=0, ammount=0;k2<a2_size;)
	{
		a2[k2]=a1[k]>>-start, a2[k2]&=mask, ++k;
		for(ammount=start+coeff_size;ammount<logbeta;ammount+=coeff_size, ++k)
			a2[k2]|=a1[k]<<ammount, a2[k2]&=mask;
		k-=ammount>logbeta;
		++k2, start=ammount-logbeta-coeff_size, start+=coeff_size&-(start<=-coeff_size);
	}
	_aligned_free(a1);
	return (short*)a2;//*/
	int a2_size=a1_size*logbeta/coeff_size;
	unsigned short *a2=(unsigned short*)_aligned_malloc(a2_size*sizeof(unsigned short), align);
	short mask=(1<<coeff_size)-1;
	for(int k=0, k2=0, start=0, ammount=0;k2<a2_size;)
	{
		a2[k2]=a1[k]>>-start, a2[k2]&=mask, ++k;
		for(ammount=start+logbeta;ammount<coeff_size;ammount+=logbeta, ++k)
			a2[k2]|=a1[k]<<ammount, a2[k2]&=mask;
		k-=ammount>coeff_size;
		++k2, start=ammount-coeff_size-logbeta, start+=logbeta&-(start<=-logbeta);
	}
//	_aligned_free(a1);
	return (short*)a2;
}
void		generate_uniform(int size_in_bytes, unsigned char *buffer)
{
//	if(use_rand)
	{
	//	short *sbuf=(short*)buffer;
		for(int k=0;k<size_in_bytes;++k)
			buffer[k]=(unsigned char)rand();
		//	buffer[k]=k+1;//
		//	buffer[k]=1;//
		//	buffer[k]=0;//
	}
	//else
	//{
	//	int success=CryptGenRandom(hProv, size_in_bytes, buffer);
	//	if(!success)
	//	{
	//		int error=GetLastError();
	//		std::cout<<"Error: CryptGenRandom(): "<<error<<endl;
	//	}
	////	memset(buffer, 0, size_in_bytes);//
	//}
}
void		newhope_convert_binomial_16(int *src, short *dst, int n)//NewHope psi_k, k=16;
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i hm1=_mm256_set1_epi32(0x55555555), hm2=_mm256_set1_epi32(0x33333333), hm3=_mm256_set1_epi32(0x0F0F0F0F), hm4=_mm256_set1_epi32(0x00FF00FF);
	for(int k=0;k<n;k+=16)
	{
		__m256i vk1=_mm256_load_si256((__m256i*)(src+k)), vk2=_mm256_load_si256((__m256i*)(src+k+8));
		__m256i t1=_mm256_and_si256(vk1, hm1), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 1), hm1);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm2), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 2), hm2);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm3), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 4), hm3);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm4), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 8), hm4);
		vk1=_mm256_add_epi16(t1, t2);

		__m256i t3=_mm256_and_si256(vk2, hm1), t4=_mm256_and_si256(_mm256_srli_epi16(vk2, 1), hm1);
		vk2=_mm256_add_epi16(t3, t4);
		t3=_mm256_and_si256(vk2, hm2), t4=_mm256_and_si256(_mm256_srli_epi16(vk2, 2), hm2);
		vk2=_mm256_add_epi16(t3, t4);
		t3=_mm256_and_si256(vk2, hm3), t4=_mm256_and_si256(_mm256_srli_epi16(vk2, 4), hm3);
		vk2=_mm256_add_epi16(t3, t4);
		t3=_mm256_and_si256(vk2, hm4), t4=_mm256_and_si256(_mm256_srli_epi16(vk2, 8), hm4);
		vk2=_mm256_add_epi16(t3, t4);

		vk1=_mm256_sub_epi16(vk1, vk2);
		_mm256_store_si256((__m256i*)(dst+k), vk1);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i hm1=_mm_set1_epi32(0x55555555), hm2=_mm_set1_epi32(0x33333333), hm3=_mm_set1_epi32(0x0F0F0F0F), hm4=_mm_set1_epi32(0x00FF00FF);
	for(int k=0;k<n;k+=8)
	{
		__m128i vk1=_mm_load_si128((__m128i*)(src+k)), vk2=_mm_load_si128((__m128i*)(src+k+4));
		__m128i t1=_mm_and_si128(vk1, hm1), t2=_mm_and_si128(_mm_srli_epi16(vk1, 1), hm1);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm2), t2=_mm_and_si128(_mm_srli_epi16(vk1, 2), hm2);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm3), t2=_mm_and_si128(_mm_srli_epi16(vk1, 4), hm3);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm4), t2=_mm_and_si128(_mm_srli_epi16(vk1, 8), hm4);
		vk1=_mm_add_epi16(t1, t2);

		__m128i t3=_mm_and_si128(vk2, hm1), t4=_mm_and_si128(_mm_srli_epi16(vk2, 1), hm1);
		vk2=_mm_add_epi16(t3, t4);
		t3=_mm_and_si128(vk2, hm2), t4=_mm_and_si128(_mm_srli_epi16(vk2, 2), hm2);
		vk2=_mm_add_epi16(t3, t4);
		t3=_mm_and_si128(vk2, hm3), t4=_mm_and_si128(_mm_srli_epi16(vk2, 4), hm3);
		vk2=_mm_add_epi16(t3, t4);
		t3=_mm_and_si128(vk2, hm4), t4=_mm_and_si128(_mm_srli_epi16(vk2, 8), hm4);
		vk2=_mm_add_epi16(t3, t4);

		vk1=_mm_sub_epi16(vk1, vk2);
		_mm_store_si128((__m128i*)(dst+k), vk1);
	}
#else
//	int *temp=new int[n];
//	generate_uniform(n*sizeof(int), (unsigned char*)temp);
	unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	for(int k=0;k<n;++k)//NewHope phi_16
	{
		auto pm=hamming_masks;
		unsigned a=src[k];//32bit
		a=(a&*pm)+(a>>1&*pm), ++pm;
		a=(a&*pm)+(a>>2&*pm), ++pm;
		a=(a&*pm)+(a>>4&*pm), ++pm;
		a=(a&*pm)+(a>>8&*pm);
		dst[k]=((short*)&a)[0]-((short*)&a)[1];
	}
//	delete[] temp;
#endif
}
void		newhope_generate_binomial_16(short *v, int n)//NewHope psi_k, k=16;
{
	const int align=sizeof(SIMD_type);
	int *temp=(int*)_aligned_malloc(n*sizeof(int), align);
//	int *temp=new int[n];
	unsigned char seed[16];
	generate_uniform(16, seed);
	FIPS202_SHAKE128(seed, 16, (unsigned char*)temp, n*sizeof(short));
//	generate_uniform(n*sizeof(int), (unsigned char*)temp);
//	generate_uniform(n*sizeof(short), (unsigned char*)v);//
	newhope_convert_binomial_16(temp, v, n);
/*	unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	for(int k=0;k<n;++k)//NewHope phi_16
	{
		auto pm=hamming_masks;
		unsigned a=temp[k];//32bit
	//	unsigned a=v[k];
		a=(a&*pm)+(a>>1&*pm), ++pm;
		a=(a&*pm)+(a>>2&*pm), ++pm;
		a=(a&*pm)+(a>>4&*pm), ++pm;
		a=(a&*pm)+(a>>8&*pm);
		//a=(a&pm[0])+(a>>1&pm[0]);
		//a=(a&pm[1])+(a>>2&pm[1]);
		//a=(a&pm[2])+(a>>4&pm[2]);
		//a=(a&pm[3])+(a>>8&pm[3]);
		v[k]=((short*)&a)[0]-((short*)&a)[1];
	}//*/
	_aligned_free(temp);
//	delete[] temp;
}
void		kyber_convert_binomial_4(short *v, int n)//Kyber binomial_4
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i ch_mask=_mm256_set1_epi16(0x00FF);
	const __m256i hm1=_mm256_set1_epi32(0x55555555), hm2=_mm256_set1_epi32(0x33333333), hm3=_mm256_set1_epi32(0x0F0F0F0F);
	for(int k=0;k<n;k+=16)
	{
		__m256i vk1=_mm256_load_si256((__m256i*)(v+k));
		vk1=_mm256_and_si256(vk1, ch_mask);
		__m256i t1=_mm256_and_si256(vk1, hm1), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 1), hm1);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm2), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 2), hm2);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm3), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 4), hm3);
		vk1=_mm256_sub_epi16(t1, t2);
		_mm256_store_si256((__m256i*)(v+k), vk1);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i ch_mask=_mm_set1_epi16(0x00FF);
	const __m128i hm1=_mm_set1_epi32(0x55555555), hm2=_mm_set1_epi32(0x33333333), hm3=_mm_set1_epi32(0x0F0F0F0F);
	for(int k=0;k<n;k+=8)
	{
		__m128i vk1=_mm_load_si128((__m128i*)(v+k));
		vk1=_mm_and_si128(vk1, ch_mask);
		__m128i t1=_mm_and_si128(vk1, hm1), t2=_mm_and_si128(_mm_srli_epi16(vk1, 1), hm1);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm2), t2=_mm_and_si128(_mm_srli_epi16(vk1, 2), hm2);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm3), t2=_mm_and_si128(_mm_srli_epi16(vk1, 4), hm3);
		vk1=_mm_sub_epi16(t1, t2);
		_mm_store_si128((__m128i*)(v+k), vk1);
	}
#else
	const unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	for(int k=0;k<n;++k)
	{
		auto pm=hamming_masks;
		unsigned a=*(char*)&v[k];//8bit
		a=(a&*pm)+(a>>1&*pm), ++pm;
		a=(a&*pm)+(a>>2&*pm), ++pm;
		v[k]=(a&*pm)-(a>>4&*pm);
	}
#endif
}
void		kyber_generate_binomial_4(short *v, int n)//Kyber binomial_4
{
	generate_uniform(n*sizeof(short), (unsigned char*)v);
	kyber_convert_binomial_4(v, n);
	//unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	//for(int k=0;k<n;++k)
	//{
	//	auto pm=hamming_masks;
	//	unsigned a=*(char*)&v[k];//8bit
	//	a=(a&*pm)+(a>>1&*pm), ++pm;
	//	a=(a&*pm)+(a>>2&*pm), ++pm;
	//	v[k]=(a&*pm)-(a>>4&*pm);
	//}
}
void		saber_convert_binomial_8(short *v, int n)
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i hm1=_mm256_set1_epi32(0x55555555), hm2=_mm256_set1_epi32(0x33333333), hm3=_mm256_set1_epi32(0x0F0F0F0F), hm4=_mm256_set1_epi32(0x00FF00FF);
	for(int k=0;k<n;k+=16)
	{
		__m256i vk1=_mm256_loadu_si256((__m256i*)(v+k));
		__m256i t1=_mm256_and_si256(vk1, hm1), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 1), hm1);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm2), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 2), hm2);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm3), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 4), hm3);
		vk1=_mm256_add_epi16(t1, t2);
		t1=_mm256_and_si256(vk1, hm4), t2=_mm256_and_si256(_mm256_srli_epi16(vk1, 8), hm4);
		vk1=_mm256_sub_epi16(t1, t2);
		_mm256_storeu_si256((__m256i*)(v+k), vk1);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i hm1=_mm_set1_epi32(0x55555555), hm2=_mm_set1_epi32(0x33333333), hm3=_mm_set1_epi32(0x0F0F0F0F), hm4=_mm_set1_epi32(0x00FF00FF);
	for(int k=0;k<n;k+=8)
	{
		__m128i vk1=_mm_loadu_si128((__m128i*)(v+k));
		__m128i t1=_mm_and_si128(vk1, hm1), t2=_mm_and_si128(_mm_srli_epi16(vk1, 1), hm1);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm2), t2=_mm_and_si128(_mm_srli_epi16(vk1, 2), hm2);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm3), t2=_mm_and_si128(_mm_srli_epi16(vk1, 4), hm3);
		vk1=_mm_add_epi16(t1, t2);
		t1=_mm_and_si128(vk1, hm4), t2=_mm_and_si128(_mm_srli_epi16(vk1, 8), hm4);
		vk1=_mm_sub_epi16(t1, t2);
		_mm_storeu_si128((__m128i*)(v+k), vk1);
	}
#else
//	generate_uniform(n*sizeof(short), (unsigned char*)v);
	unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	for(int k=0;k<n;++k)
	{
		auto pm=hamming_masks;
		unsigned short a=v[k];//16bit
		a=(a&*pm)+(a>>1&*pm), ++pm;
		a=(a&*pm)+(a>>2&*pm), ++pm;
		a=(a&*pm)+(a>>4&*pm);
		v[k]=((char*)&a)[0]-((char*)&a)[1];
	}
#endif
}
void		saber_generate_binomial_8(short *v, int n)
{
	generate_uniform(n*sizeof(short), (unsigned char*)v);
	saber_convert_binomial_8(v, n);
/*	unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	for(int k=0;k<n;++k)
	{
		auto pm=hamming_masks;
		unsigned short a=v[k];//16bit
		a=(a&*pm)+(a>>1&*pm), ++pm;
		a=(a&*pm)+(a>>2&*pm), ++pm;
		a=(a&*pm)+(a>>4&*pm);
		v[k]=((char*)&a)[0]-((char*)&a)[1];
	}//*/
}
void		kyber_uniform_rejection_sampling(short *out, const int out_size, const unsigned char *in, const int in_size, const int q)
{
/*	int logq=log_2(q), mask=(1<<(logq+1))-1;	//2nd, 5th/6th: error, different A
	short *temp=new short[out_size];
	for(int k=0, k2=out_size;k<out_size;++k)
	{
		if(k2>=out_size)
		{
			FIPS202_SHAKE128(in, in_size, (unsigned char*)temp, out_size*sizeof(short));
			k2=0;
		}
		for(;k<out_size&&k2<out_size;++k2)
		{
			auto &vk=temp[k2];
			int sign=-(vk<0);
			vk^=sign, vk-=sign;//abs
			vk&=mask;
			if(temp[k2]<q)
				out[k]=vk, ++k;
		}
	}
	delete[] temp;//*/
	FIPS202_SHAKE128(in, in_size, (unsigned char*)out, out_size*sizeof(short));
	const int barrett_m=0x10000/q;
//	const int mask=(1<<(log_2(q)+1))-1;
#if PROCESSOR_ARCH>=SSE2
#else
	for(int kx=0;kx<out_size;++kx)
	{
		auto &vk=out[kx];
	//	vk&=mask;//far from uniform
		vk-=(vk*barrett_m>>16)*q;
		vk-=q&-(vk>q);
	}
#endif
	//for(int k=0;k<out_size;++k)//
	//	std::cout<<'\t'<<out[k];
	//std::cout<<endl;
	//for(int k=0;k<out_size;++k)//
	//	out[k]=0;
}
void		kyber_compress(short *a, short n, int q, int d)
//void		kyber_compress(short *a, short n, double _2d_q)
{
/*#if PROCESSOR_ARCH>=SSE2
	const __m128i sh=_mm_set_epi32(0, 0, 0, d+1),
		mq=_mm_set1_pd(q), m2q=_mm_set1_pd(q<<1);
	for(int k=0;k<n;k+=8)
	{
		__m128i v=_mm_loadu_si128((__m128i*)(a+k));
		v=_mm_sll_epi16(v, sh);
		v=_mm_add_epi16(v, mq);
	}
#else//*/
	long long inv_q=0x400000/(q<<1);
//	double _2d_q=double(1<<d)/q;
	for(int k=0;k<n;++k)
		a[k]=short(((a[k]<<(d+1))+q)*inv_q>>22);
	//	a[k]=((a[k]<<(d+1))+q)/(q<<1);
	//	a[k]=(short)floor(a[k]*_2d_q+0.5);//round(vk*2^d/q) = floor((vk<<d)/q+0.5) = floor(((vk<<d+1)+q)/(q<<1))
//#endif
}
void		kyber_decompress(short *a, short n, int q, int d)
//void		kyber_decompress(short *a, short n, double _2d_q)
{
#if PROCESSOR_ARCH>=SSE2
	const __m128i mq=_mm_set1_epi16(q),
		sh_lo=_mm_set_epi32(0, 0, 0, d),
		sh_hi=_mm_set_epi32(0, 0, 0, 16-d);
	for(int k=0;k<n;k+=8)
	{
		__m128i v=_mm_loadu_si128((__m128i*)(a+k));
		__m128i v_lo=_mm_mullo_epi16(v, mq);
		__m128i v_hi=_mm_mulhi_epi16(v, mq);
		v_lo=_mm_srl_epi16(v_lo, sh_lo);
		v_hi=_mm_sll_epi16(v_hi, sh_hi);
		v_lo=_mm_or_si128(v_lo, v_hi);
		_mm_storeu_si128((__m128i*)(a+k), v_lo);
	}
#else
//	double q_2d=double(q)/(1<<d);
//	double q_2d=1/_2d_q;
	for(int k=0;k<n;++k)
		a[k]=a[k]*q>>d;
	//	a[k]=(short)floor(a[k]*q_2d+0.5);//round(vk*q/2^d) = floor((vk*q>>d)+0.5) = vk*q>>d
#endif
}
void		NewHope_HelpRec(short const *x, unsigned char *r)//x 1024 mod q -> r 1024 mod (2^r=4)		pages 7, 19
{
	const short q=12289;
	short v[4];
	bool b=rand()%2!=0;
	for(int k=0;k<256;++k)
	{
		auto pv=x+(k<<2);
		int manhattan=0;
		for(int k2=0;k2<4;++k2)
		{
			v[k2]=(pv[k2]<<3)+(b<<2)+12289;//v0[i]=v[k2]/12289>>1
			manhattan+=abs((q*pv[k2]<<1)-v[0]);//2q*||x-v0||1
		}
		bool _k=manhattan>=2*q;
		for(int k2=0;k2<4;++k2)
			v[k2]-=q&-(int)_k, v[k2]/=q, v[k2]>>=1;//2 bits
		auto pr=r+k;
		pr[0]=v[0]-v[3], pr[0]%=4, pr[0]+=4&-(pr[0]<0);
		pr[1]=v[1]-v[3], pr[1]%=4, pr[1]+=4&-(pr[1]<0);
		pr[2]=v[2]-v[3], pr[2]%=4, pr[2]+=4&-(pr[2]<0);
		pr[3]=(int)_k+(v[3]<<1), pr[3]%=4, pr[3]+=4&-(pr[3]<0);
	//	r[k]=(v[0]-v[3])<<6|(v[1]-v[3])<<4|(v[2]-v[3])<<2|(int)_k+(vv[3]<<1);
	}
}
void		NewHope_Rec(short const *x, unsigned char const *r, unsigned char *result)//1024 mod q, 1024 mod (2^r=4) -> 256 bit
{
	memset(result, 0, 256*sizeof(unsigned char));
	const int q=12289, amp=q<<3;//q*2^r*2, r=2
	for(int k=0;k<256;++k)
	{
		auto px=x+(k<<2);
		auto pr=r+(k<<2);
		int manhattan=0, temp;//2(x*4 - Br*q) / (amp=2*q*4) = x/q-Br/2^r
		temp=(px[0]<<3)-((pr[0]<<1)+pr[3])*q,	manhattan+=abs(temp-temp/amp*amp);
		temp=(px[1]<<3)-((pr[1]<<1)+pr[3])*q,	manhattan+=abs(temp-temp/amp*amp);
		temp=(px[2]<<3)-((pr[2]<<1)+pr[3])*q,	manhattan+=abs(temp-temp/amp*amp);
		temp=(px[3]<<3)-pr[3]*q,				manhattan+=abs(temp-temp/amp*amp);
		result[k/8]|=(manhattan>amp)<<k%8;
	}
/*	memset(result, 0, 256*sizeof(unsigned char));
	const int q=12289, amp=q<<2;//q*2^r, r=2
	for(int k=0;k<256;++k)
	{
		auto px=x+(k<<2);
		auto pr=r+(k<<2);
		int manhattan=0, temp, p3_2=pr[3]>>1;//(x*4 - Br*q) / (amp=q*4) = x/q-Br/2^r
		temp=(px[0]<<3)-(pr[0]+p3_2)*q, manhattan+=abs(temp-temp/amp*amp);
		temp=(px[1]<<3)-(pr[1]+p3_2)*q, manhattan+=abs(temp-temp/amp*amp);
		temp=(px[2]<<3)-(pr[2]+p3_2)*q, manhattan+=abs(temp-temp/amp*amp);
		temp=(px[3]<<3)-p3_2*q,			manhattan+=abs(temp-temp/amp*amp);
		result[k/8]|=(manhattan>m_mag)<<k%8;
	}//*/
/*	const int q=12289, q2r=q<<2;
	short temp[4];
	for(int k=0;k<256;++k)
	{
		auto px=x+(k<<2);
		auto pr=r+(k<<2);
		int r3_2=pr[3]>>1;//Br, B={u0, u1, u2, g}
		temp[0]=pr[0]+r3_2;
		temp[1]=pr[1]+r3_2;
		temp[2]=pr[2]+r3_2;
		temp[3]=r3_2;
		int manhattan=0;
		for(int k2=0;k2<4;++k2)
		{
			temp[k2]=(px[k2]<<2)-temp[k2]*q;
			manhattan+=abs(temp[k2]);
		}
		result[k/8]|=(manhattan>q2r)<<k%8;
	}//*/
}

void		multiply_polynomials(const short *a, const short *b, short *ab, int n, int q, short barrett_k, short barrett_m, bool anti_cyclic)//naive
{
	for(int k=0;k<n;++k)
	{
		long long sum=0;
		int sign_mask=-(int)anti_cyclic;
		for(int k2=0;k2<=k;++k2)
		{
			sum+=a[k2]*b[k-k2];
			sum-=(sum*barrett_m>>barrett_k)*q;
			//barrett_reduction_quick(sum, q, n, barrett_k, barrett_m);
		}
		for(int k2=k+1;k2<n;++k2)
		{
			sum+=(a[k2]*b[n-1-(k2-(k+1))]^sign_mask)-sign_mask;
			sum-=(sum*barrett_m>>barrett_k)*q;
			//barrett_reduction_quick(sum, q, n, barrett_k, barrett_m);
		}
		sum-=q&-(sum>q);
		sum+=q&-(sum<0);
		ab[k]=(short)sum;
	/*	auto &vk=ab[n-1-k];
		int sum=0, sign_mask=-(int)anti_cyclic;
		//vk=0;
		int range1=n-1-k;
		for(int k2=range1;k2>=0;--k2)
			sum+=a[k2]*b[range1-k2];
			//vk+=barrett_reduction_quick(a[k2]*b[range1-k2], q, n, barrett_k, barrett_m);
		//	vk+=a[k2]*b[range1-k2]%q;
		sum=barrett_reduction_quick(sum, q, n, barrett_k, barrett_m);

		for(int k2=n-1;k2>range1;--k2)
			sum-=(a[k2]*b[n-k2+range1]^sign_mask)+anti_cyclic;//-=: x^n+1
		//	vk-=(a[k2]*b[n-k2+range1]^-(int)anti_cyclic)+anti_cyclic, vk%=q, vk+=q&-(vk<0);//-=: x^n+1
		//	vk-=a[k2]*b[n-k2+range1], vk%=v2_q, vk+=v2_q&-(vk<0);//-=: x^n+1
		//	vk+=a[k2]*b[n-k2+range1], vk%=v2_q;					//+=: x^n-1
		vk=barrett_reduction(sum, q, n, barrett_k, barrett_m);//*/
	}
}
void		multiply_polynomials_mod_powof2_add(const short *a, const short *b, short *ab, int n, int logq, bool anti_cyclic)//naive
{
	int mask=(1<<logq)-1;
	for(int k=0;k<n;++k)
	{
		auto &sum=ab[k];
	//	long long sum=0;
		int sign_mask=-(int)anti_cyclic;
	//	short sign_mask=(short)anti_cyclic<<(logq-1);
		for(int k2=0;k2<=k;++k2)
		{
			sum=sum+a[k2]*b[k-k2]&mask;
			//sum=sum>=0?sum&mask:-(-sum&mask);
			//barrett_reduction_quick(sum, q, n, barrett_k, barrett_m);
		}
		for(int k2=k+1;k2<n;++k2)
		{
			sum=(sum+(a[k2]*b[n-1-(k2-(k+1))]^sign_mask)-sign_mask)&mask;
		//	sum=(sum+(a[k2]*b[n-1-(k2-(k+1))]^sign_mask))&mask;//sign-magnitude?
		//	sum=(sum+a[k2]*b[n-1-(k2-(k+1))]^sign_mask)&mask;
			//sum=sum>=0?sum&mask:-(-sum&mask);
			//barrett_reduction_quick(sum, q, n, barrett_k, barrett_m);
		}
		//sum-=q&-(sum>q);
		//sum+=q&-(sum<0);
	//	ab[k]=(short)sum;
	//	auto &vk=ab[k];
	//	vk+=(short)sum, vk=vk>=0?vk&mask:-(-vk&mask);
	/*	auto &vk=ab[n-1-k];
		int sum=0, sign_mask=-(int)anti_cyclic;
		//vk=0;
		int range1=n-1-k;
		for(int k2=range1;k2>=0;--k2)
			sum+=a[k2]*b[range1-k2];
			//vk+=barrett_reduction_quick(a[k2]*b[range1-k2], q, n, barrett_k, barrett_m);
		//	vk+=a[k2]*b[range1-k2]%q;
		sum=barrett_reduction_quick(sum, q, n, barrett_k, barrett_m);

		for(int k2=n-1;k2>range1;--k2)
			sum-=(a[k2]*b[n-k2+range1]^sign_mask)+anti_cyclic;//-=: x^n+1
		//	vk-=(a[k2]*b[n-k2+range1]^-(int)anti_cyclic)+anti_cyclic, vk%=q, vk+=q&-(vk<0);//-=: x^n+1
		//	vk-=a[k2]*b[n-k2+range1], vk%=v2_q, vk+=v2_q&-(vk<0);//-=: x^n+1
		//	vk+=a[k2]*b[n-k2+range1], vk%=v2_q;					//+=: x^n-1
		vk=barrett_reduction(sum, q, n, barrett_k, barrett_m);//*/
	}
}
/*		NTT multiplication
	i7 6800k broadwell-e
	r2 avx2					 3.86  12833c	//7.88 26180c
	r2 sse i				
	r2 sse i avx			
	r2 sse i avx2			
	r2 sse					 7.97  26476c	//9.11 30251c
	r4 avx2					 9.32  30937c	//12.97 43072c
	r4 sse vs				16.90  56125c	//19.00 63073c
	r4 sse intel sse3		
	r4 intel sse3			
	r2						49.08 162968c	//136.4 453036c
	r4						68.68 228047c	//128.4 426303c

	i5 2410M intel 2018 arch:AVX
	r2 sse					
	r2 sse arch:SSE2		
	r4 sse					
	r4 sse arch:SSE2		
	r2						
	r4						

	i5 430m intel 2018
	r2 sse					 35.56  78507c	//33.46  73879c	//34.68  76573c	//39.31  86788c
	r4 sse					 51.70 114150c	//54.08 119406c
	r4						323.6  714532c
	r2						337.3  744856c

	i5 430m VS2013
	r2 sse					 32.62  72026c	31.93  70508c	//32.95  72755c	//35.35  78060c	//41.71  92089c
	r4 sse					 54.29 119876c	52.86 116718c	//56.50 124761c
	r4						339.9  750429c
	r2						356.6  787300c

	c2d u7700 VCE2010
	r2 sse					
	r4 sse					
	r4						
	r2						
	//*/
/*			64		256		1024		ntt test
	i7 6800k broadwell-e
	r2 avx2					 2.75  9124c	//5.85 19411c	//6.42 21316c	//6.78 22498c	//8.6 29Kc?
	r2 sse i				 7.02 23332c
	r2 sse i avx			 7.76 25769c
	r2 sse i avx2			 7.99 26542c
	r2 sse					 5.30 17607c	//10 33Kc
	r4 avx2					 7.11 23613c	//12.14 40307c	//8.44 28Kc	//9.74 36Kc	//52 54 56 60
	r4 sse vs				10.98 36455c	//12 41Kc		//14			//57 61 62 63 66
	r4 sse intel sse3		49 54 55 56
	r4 intel sse3			68
	r4						44.24 146875c	//86.99 288845c
	r2						34.27 113802c	//96.6 320819c

	i5 2410M intel 2018 arch:AVX
	r2 sse					 1.55 22Kc	//1.9  27Kc
	r2 sse arch:SSE2		 1.70 24Kc	//2.2  32Kc
	r4 sse					 2.4  34Kc	//41 42	//56 57	//141
	r4 sse arch:SSE2		 2.48 36Kc	//46 47
	r2		4.5 7.7	22 24	98 99
	r4		4.5 7	23 26	109

	i5 430m intel 2018
	r2 sse	15		56		 18.34 40493c	//25 56Kc	//26 57Kc	//33 73Kc	//37  82Kc	//72 159Kc, 80
	r4 sse	15		56		 36 80Kc	//81 178Kc		//169 372c, 172 380c	//200	//209	//288 272 267	//233 //379
	r4 sse3s				172 380c
	r2		15		70		143.31 316434c	//296
	r4		13		67		177.82 392619c	//274

	i5 430m VS2013
	r2 sse4					 18.65 41179c	//20.42 45106c	//20.03 44224c	//22.62 49948c	//18.84X 41608c	//21.8 48034c	//23.4ms 51688c	//25  56Kc	//32  70Kc	//37  82Kc	//84 186Kc
	r4 sse4	17		87		 33.48 73923c	//37  82Kc	//47 103Kc	//50 111Kc	//55 122Kc	//65 144Kc	//82 181Kc	//119		//146 322c, 147 324c	//158ms/349cyc	//156 159 162	//218	//223 222	//244 227 225	//297 291	//265	//266	//259	//297	//348	//385	//408
	r4 sse3s				173 381Kc		//236	//286	//326
	r2		15		78 72	144.02 317985c	//213.86 472209c	//265 586Kc	//327 718Kc			//340
	r4		15		74		186.45 411684c	//231.18 510438c	//227	502Kc	//324 717Kc	//307 678Kc	//327

	c2d u7700 VCE2010
	r2 sse					 20.70  26892c	//21.20 27547c	//20.86 27102c 21.93 28481c	//26 37Kc	//39.3ms 51013c (32)	//76 98Kc
	r4 sse3s 14ms	104ms	 47.66  61906c	//50 65Kc	//217 282c	//282	//343	//347	//509
	r2						127.89 166103c	//454
	r4		10		84		210.21 273033c	//385
	//*/

inline void	subtract_polynomials(short *dst, const short *a, const short *b, short n, short q)
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i m_q=_mm256_set1_epi16(q);
	for(int k=0;k<n;k+=16)
	{
		__m256i va=_mm256_loadu_si256((__m256i*)(a+k));
		__m256i vb=_mm256_loadu_si256((__m256i*)(b+k));
		va=_mm256_sub_epi16(va, vb);
		vb=_mm256_cmpgt_epi16(va, m_q);
		vb=_mm256_and_si256(vb, m_q);
		va=_mm256_sub_epi16(va, vb);
		_mm256_storeu_si256((__m256i*)(dst+k), va);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i m_q=_mm_set1_epi16(q);
	for(int k=0;k<n;k+=8)
	{
		__m128i va=_mm_loadu_si128((__m128i*)(a+k));
		__m128i vb=_mm_loadu_si128((__m128i*)(b+k));
		va=_mm_sub_epi16(va, vb);
		vb=_mm_cmplt_epi16(m_q, va);
		vb=_mm_and_si128(vb, m_q);
		va=_mm_sub_epi16(va, vb);
		_mm_storeu_si128((__m128i*)(dst+k), va);
	}
#else
	for(int k=0;k<n;++k)
	{
		auto &vk=dst[k];
		vk=a[k]-b[k], vk+=q&-(vk<0);
	}
#endif
}
inline void	add_polynomials(short *dst, const short *a, const short *b, short n, short q)
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i m_q=_mm256_set1_epi16(q);
	for(int k=0;k<n;k+=16)
	{
		__m256i va=_mm256_loadu_si256((__m256i*)(a+k));
		__m256i vb=_mm256_loadu_si256((__m256i*)(b+k));
		va=_mm256_add_epi16(va, vb);
		vb=_mm256_cmpgt_epi16(va, m_q);
		vb=_mm256_and_si256(vb, m_q);
		va=_mm256_sub_epi16(va, vb);
		_mm256_storeu_si256((__m256i*)(dst+k), va);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i m_q=_mm_set1_epi16(q);
	for(int k=0;k<n;k+=8)
	{
		__m128i va=_mm_loadu_si128((__m128i*)(a+k));
		__m128i vb=_mm_loadu_si128((__m128i*)(b+k));
		va=_mm_add_epi16(va, vb);
		vb=_mm_cmplt_epi16(m_q, va);
		vb=_mm_and_si128(vb, m_q);
		va=_mm_sub_epi16(va, vb);
		_mm_storeu_si128((__m128i*)(dst+k), va);
	}
#else
	for(int k=0;k<n;++k)
	{
		auto &vk=dst[k];
		vk=a[k]+b[k], vk-=q&-(vk>q);
	}
#endif
}
void		make_small(short *a, int n, short q, short q_1)//a3 = a1 * a2
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i m_q=_mm256_set1_epi16(q);
	const __m256i m_q_1=_mm256_set1_epi16(q_1);
	const __m256i m_beta=_mm256_set1_epi16(0x10000%q);
	const __m256i m_q_2=_mm256_set1_epi16(q>>1);
	for(int k=0;k<n;k+=16)
	{
		__m256i va=_mm256_load_si256((__m256i*)(a+k));
		__m256i cmp_mask=_mm256_cmpgt_epi16(va, m_q_2);
		cmp_mask=_mm256_and_si256(cmp_mask, m_q);
		va=_mm256_sub_epi16(va, m_q);
		_mm256_store_si256((__m256i*)(a+k), va);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i m_q=_mm_set1_epi16(q);
	const __m128i m_q_1=_mm_set1_epi16(q_1);
	const __m128i m_beta=_mm_set1_epi16(0x10000%q);
	const __m128i m_q_2=_mm_set1_epi16(q>>1);
	for(int k=0;k<n;k+=8)
	{
		__m128i va=_mm_load_si128((__m128i*)(a+k));
		__m128i cmp_mask=_mm_cmpgt_epi16(va, m_q_2);
		cmp_mask=_mm_and_si128(cmp_mask, m_q);
		va=_mm_sub_epi16(va, m_q);
		_mm_store_si128((__m128i*)(a+k), va);
	}
#else
	const int amp=q/2;
	for(int k=0;k<n;++k)
	{
		auto &vk=a[k];
	//	int temp=vk*4091;
	//	vk=*((short*)&temp)*q_1, vk=vk*q>>16, vk=((short*)&temp)[1]-vk;//montgomery reduction
		vk-=q&-(vk>amp);
	}
#endif
}
void		multiply_ntt_add(short *dst, short const *a1, short const *a2, NTT_params const &p)
{
#if PROCESSOR_ARCH>=AVX2
	auto &n=p.n, &q=p.q, &q_1=p.q_1, &beta_q=p.beta_q;
	short beta2=beta_q*beta_q%q;
	const __m256i m_q=_mm256_set1_epi16(q);
	const __m256i m_q_1=_mm256_set1_epi16(q_1);
	const __m256i m_beta2=_mm256_set1_epi16(beta2);
	for(int k=0;k<n;k+=16)
	{
		__m256i va=_mm256_load_si256((__m256i*)(a1+k));
		__m256i vb=_mm256_load_si256((__m256i*)(a2+k));
		__m256i vc=_mm256_load_si256((__m256i*)(dst+k));
		__m256i v_lo=_mm256_mullo_epi16(va, vb);
		__m256i v_hi=_mm256_mulhi_epi16(va, vb);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		v_lo=_mm256_sub_epi16(v_hi, v_lo);
		va=_mm256_mullo_epi16(v_lo, m_beta2);
		vb=_mm256_mulhi_epi16(v_lo, m_beta2);
		va=_mm256_mullo_epi16(va, m_q_1);
		va=_mm256_mulhi_epi16(va, m_q);
		va=_mm256_sub_epi16(vb, va);

		va=_mm256_add_epi16(va, vc);
		v_lo=_mm256_cmpgt_epi16(va, m_q);
		v_lo=_mm256_and_si256(v_lo, m_q);
		va=_mm256_sub_epi16(va, v_lo);
		_mm256_store_si256((__m256i*)(dst+k), va);
	}
	_m_empty();
#elif PROCESSOR_ARCH>=SSE2
	auto &n=p.n, &q=p.q, &q_1=p.q_1, &beta_q=p.beta_q;
	short beta2=beta_q*beta_q%q;
	const __m128i m_q=_mm_set1_epi16(q);
	const __m128i m_q_1=_mm_set1_epi16(q_1);
	const __m128i m_beta2=_mm_set1_epi16(beta2);
	for(int k=0;k<n;k+=8)
	{
		__m128i va=_mm_load_si128((__m128i*)(a1+k));
		__m128i vb=_mm_load_si128((__m128i*)(a2+k));
		__m128i vc=_mm_load_si128((__m128i*)(dst+k));
		__m128i v_lo=_mm_mullo_epi16(va, vb);
		__m128i v_hi=_mm_mulhi_epi16(va, vb);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);//montgomery reduction 1
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		v_lo=_mm_sub_epi16(v_hi, v_lo);

		va=_mm_mullo_epi16(v_lo, m_beta2);
		vb=_mm_mulhi_epi16(v_lo, m_beta2);
		va=_mm_mullo_epi16(va, m_q_1);//montgomery reduction 2
		va=_mm_mulhi_epi16(va, m_q);
		va=_mm_sub_epi16(vb, va);

		va=_mm_add_epi16(va, vc);
		v_lo=_mm_cmplt_epi16(m_q, va);
		v_lo=_mm_and_si128(v_lo, m_q);
		va=_mm_sub_epi16(va, v_lo);
		_mm_store_si128((__m128i*)(dst+k), va);
	}
	_m_empty();
#else
	auto &n=p.n, &q=p.q, &q_1=p.q_1, &beta_q=p.beta_q;
	short beta2=beta_q*beta_q%q;
	for(int k=0;k<n;++k)
	{
		auto &vk=dst[k];
		int temp=a1[k]*a2[k];
		short vk2=short(*(short*)&temp*q_1); vk2=vk2*q>>16, vk2=((short*)&temp)[1]-vk2;
		temp=vk2*beta2;
		vk2=short(*(short*)&temp*q_1), vk2=vk2*q>>16, vk2=((short*)&temp)[1]-vk2;
		vk+=vk2, vk-=q&-(vk>q);

		//vk+=a1[k]*a2[k]%q, vk+=(q&-(vk<0))-(q&-(vk>q));
	}
	//	dst[k]+=a1[k]*a2[k]%q, dst[k]+=q&-(dst[k]<0);
#endif
}
void		multiply_ntt(short *dst, short const *a1, short const *a2, NTT_params const &p)
{
#if PROCESSOR_ARCH>=AVX2
	auto &n=p.n, &q=p.q, &q_1=p.q_1, &beta_q=p.beta_q;
	short beta2=beta_q*beta_q%q;
	const __m256i m_q=_mm256_set1_epi16(q);
	const __m256i m_q_1=_mm256_set1_epi16(q_1);
	const __m256i m_beta2=_mm256_set1_epi16(beta2);
	for(int k=0;k<n;k+=16)
	{
		__m256i va=_mm256_load_si256((__m256i*)(a1+k));
		__m256i vb=_mm256_load_si256((__m256i*)(a2+k));
		__m256i v_lo=_mm256_mullo_epi16(va, vb);
		__m256i v_hi=_mm256_mulhi_epi16(va, vb);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		v_lo=_mm256_sub_epi16(v_hi, v_lo);
		va=_mm256_mullo_epi16(v_lo, m_beta2);
		vb=_mm256_mulhi_epi16(v_lo, m_beta2);
		va=_mm256_mullo_epi16(va, m_q_1);
		va=_mm256_mulhi_epi16(va, m_q);
		va=_mm256_sub_epi16(vb, va);
		_mm256_store_si256((__m256i*)(dst+k), va);
	}
	_m_empty();
#elif PROCESSOR_ARCH>=SSE2
	auto &n=p.n, &q=p.q, &q_1=p.q_1, &beta_q=p.beta_q;
	short beta2=beta_q*beta_q%q;
	const __m128i m_q=_mm_set1_epi16(q);
	const __m128i m_q_1=_mm_set1_epi16(q_1);
	const __m128i m_beta2=_mm_set1_epi16(beta2);
	for(int k=0;k<n;k+=8)//34.67 76567c
	{
		__m128i va=_mm_load_si128((__m128i*)(a1+k));
		__m128i vb=_mm_load_si128((__m128i*)(a2+k));
		__m128i v_lo=_mm_mullo_epi16(va, vb);
		__m128i v_hi=_mm_mulhi_epi16(va, vb);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);//montgomery reduction 1
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		v_lo=_mm_sub_epi16(v_hi, v_lo);
		va=_mm_mullo_epi16(v_lo, m_beta2);
		vb=_mm_mulhi_epi16(v_lo, m_beta2);
		va=_mm_mullo_epi16(va, m_q_1);//montgomery reduction 2
		va=_mm_mulhi_epi16(va, m_q);
		va=_mm_sub_epi16(vb, va);
		_mm_store_si128((__m128i*)(dst+k), va);
	}//*/
/*	for(int k=0;k<n;k+=16)//35.51 78425c
	{
		__m128i va1=_mm_load_si128((__m128i*)(a1+k));
		__m128i va2=_mm_load_si128((__m128i*)(a1+k+8));
		__m128i vb1=_mm_load_si128((__m128i*)(a2+k));
		__m128i vb2=_mm_load_si128((__m128i*)(a2+k+8));

		__m128i v_lo=_mm_mullo_epi16(va1, vb1);
		__m128i v_hi=_mm_mulhi_epi16(va1, vb1);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		v_lo=_mm_sub_epi16(v_hi, v_lo);
		va1=_mm_mullo_epi16(v_lo, m_beta2);
		vb1=_mm_mulhi_epi16(v_lo, m_beta2);
		va1=_mm_mullo_epi16(va1, m_q_1);
		va1=_mm_mulhi_epi16(va1, m_q);
		va1=_mm_sub_epi16(vb1, va1);

		v_lo=_mm_mullo_epi16(va2, vb2);
		v_hi=_mm_mulhi_epi16(va2, vb2);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		v_lo=_mm_sub_epi16(v_hi, v_lo);
		va2=_mm_mullo_epi16(v_lo, m_beta2);
		vb2=_mm_mulhi_epi16(v_lo, m_beta2);
		va2=_mm_mullo_epi16(va2, m_q_1);
		va2=_mm_mulhi_epi16(va2, m_q);
		va2=_mm_sub_epi16(vb2, va2);
		_mm_store_si128((__m128i*)(dst+k), va1);
		_mm_store_si128((__m128i*)(dst+k+8), va2);
	}//*/
	_m_empty();
#else
	auto &n=p.n, &q=p.q, &q_1=p.q_1, &beta_q=p.beta_q;
	short beta2=beta_q*beta_q%q;
	for(int k=0;k<n;++k)
	{
		auto &vk=dst[k];
		int temp=a1[k]*a2[k];
		vk=short(*(short*)&temp*q_1), vk=vk*q>>16, vk=((short*)&temp)[1]-vk;
		temp=vk*beta2;
		vk=short(*(short*)&temp*q_1), vk=vk*q>>16, vk=((short*)&temp)[1]-vk;

		//vk=a1[k]*a2[k]%q, vk+=(q&-(vk<0))-(q&-(vk>q));
	}
	//	dst[k]=a1[k]*a2[k]%q, dst[k]+=q&-(dst[k]<0);
#endif
}

short		bitreverse_table[1024]={0};
void		bitreverse_init(int n, int logn)
{
	for(int k=0;k<n;++k)
	{
		int k2=0;
//#ifdef R4_SSE
//		for(int k3=0, temp=k;k3<logn;k3+=2)//2 bit reverse: pure radix 4
//			k2<<=2, k2|=temp&3, temp>>=2;
//#elif defined R2_SSE
		for(int k3=0, temp=k;k3<logn;++k3)//radix 2
			k2<<=1, k2|=temp&1, temp>>=1;
//#endif
		bitreverse_table[k]=k2;
	}
}
void		number_transform_initialize_begin(short n, short q, short &n_1, short &beta_q, short &q_1)
{
	int inv=0;
	extended_euclidean_algorithm(n, q, inv);
	n_1=inv;
	beta_q=0x10000%q;
	extended_euclidean_algorithm(q, 0x10000, inv);
	q_1=inv;
}
#if PROCESSOR_ARCH>=AVX2
//struct NTT_params_AVX2
//{
//	short q, n, n_1, q_1, beta_q;
//	__m256i *m_phi, *m_iphi, *m_r, *m_ir, *m_stage, *m_istage;
//};
void		number_transform_initialize_avx2(short n, short q, short w, short sqrt_w, bool anti_cyclic, NTT_params_AVX2 &p)
{
	p.n=n, p.q=q, p.w=w, p.sqrt_w=sqrt_w, p.sbar_m=0x10000/q, p.anti_cyclic=anti_cyclic;
	number_transform_initialize_begin(n, q, p.n_1, p.beta_q, p.q_1);
	bitreverse_init(n, log_2(n));
	short *roots=(short*)_aligned_malloc(n*sizeof(short), sizeof(__m256i)), *iroots=(short*)_aligned_malloc(n*sizeof(short), sizeof(__m256i));
	for(int k=0, wk=1;k<n;++k)//initialize NTT roots
		roots[k]=wk, iroots[(2*n-k)%n]=wk, wk*=w, wk%=q;
	
	int n_16=n>>4;
	p.m_phi=(__m256i*)_aligned_malloc(n_16*sizeof(__m256i), sizeof(__m256i));
	p.m_iphi=(__m256i*)_aligned_malloc(n_16*sizeof(__m256i), sizeof(__m256i));

	__m256i m_q=_mm256_set1_epi16(q);
	__m256i m_q_1=_mm256_set1_epi16(p.q_1);
	__m256i m_zero=_mm256_setzero_si256();
	__m256i m_beta2=_mm256_set1_epi16(p.beta_q*p.beta_q%q);//r2	10952*		1: 4091, 2: 10952, 3: 11227, 4: 5664 (stg4), 5: 6659, 6: 9545 (6), 7: 6442, 8: 6606 (8), 9: 1635, 10: 3569 (10), 11: 1447, 12: 8668, 13: 7023, 14: 11700, 15: 11334
	{
		short *t_phi=(short*)malloc(n*sizeof(short));
#ifdef IA32_USE_MONTGOMERY_REDUCTION
		int factor=p.beta_q;
		int f2=p.sqrt_w*p.beta_q%q;
#else
		int factor=p.beta_q;//%
#endif
		for(int k=0;k<n;++k)//{1, phi, ..., phi^(n-1)}	//218ms
		{
			t_phi[k]=factor;
#ifdef IA32_USE_MONTGOMERY_REDUCTION
			int pr=factor*f2;	factor=short(*(short*)&pr*p.q_1), factor=factor*q>>16, factor=((short*)&pr)[1]-factor;//, factor+=q&-(factor<0);
#else
			factor=factor*p.sqrt_w%q, factor+=q&-(factor<0);
#endif
		}
		auto p_phi=(short*)p.m_phi;
		for(int k=0;k<n;++k)
			p_phi[k]=t_phi[bitreverse_table[k]];
		free(t_phi);//*/
		
		__m256i m_factor=_mm256_set1_epi16(iroots[n-8]*p.beta_q%q);
		__m256i m_wpowers=_mm256_setr_epi16(
			(q-iroots[n-8])*p.n_1%q, (q-iroots[n-7]*sqrt_w%q)*p.n_1%q, (q-iroots[n-7])*p.n_1%q, (q-iroots[n-6]*sqrt_w%q)*p.n_1%q,
			(q-iroots[n-6])*p.n_1%q, (q-iroots[n-5]*sqrt_w%q)*p.n_1%q, (q-iroots[n-5])*p.n_1%q, (q-iroots[n-4]*sqrt_w%q)*p.n_1%q,
			(q-iroots[n-4])*p.n_1%q, (q-iroots[n-3]*sqrt_w%q)*p.n_1%q, (q-iroots[n-3])*p.n_1%q, (q-iroots[n-2]*sqrt_w%q)*p.n_1%q,
			(q-iroots[n-2])*p.n_1%q, (q-iroots[n-1]*sqrt_w%q)*p.n_1%q, (q-iroots[n-1])*p.n_1%q, (q-sqrt_w)*p.n_1%q);

		__m256i m_powers_lo=_mm256_mullo_epi16(m_wpowers, m_beta2);
		m_wpowers=_mm256_mulhi_epi16(m_wpowers, m_beta2);
		m_powers_lo=_mm256_mullo_epi16(m_powers_lo, m_q_1);
		m_powers_lo=_mm256_mulhi_epi16(m_powers_lo, m_q);
		m_wpowers=_mm256_sub_epi16(m_wpowers, m_powers_lo);
		//m_factor=_mm256_set1_epi16(iroots[n-8]*p.beta_q%q);
		_mm256_store_si256(p.m_iphi+n_16-1, m_wpowers);
		for(int k=n_16-2;k>=0;--k)
		{
			__m256i m_powers_lo=_mm256_mullo_epi16(m_wpowers, m_factor);
			m_wpowers=_mm256_mulhi_epi16(m_wpowers, m_factor);
			m_powers_lo=_mm256_mullo_epi16(m_powers_lo, m_q_1);
			m_powers_lo=_mm256_mulhi_epi16(m_powers_lo, m_q);
			m_wpowers=_mm256_sub_epi16(m_wpowers, m_powers_lo);
			_mm256_store_si256(p.m_iphi+k, m_wpowers);
			//print_register(p.m_iphi[k], q);//
		}
	}

	p.m_r=(__m256i*)_aligned_malloc(n_16*sizeof(__m256i), sizeof(__m256i));
	p.m_ir=(__m256i*)_aligned_malloc(n_16*sizeof(__m256i), sizeof(__m256i));
	for(int rstep=n>>8, rstep16=rstep<<4, m_k=0;rstep>0;rstep>>=1, rstep16>>=1)
	{
		for(int j=0, n_2=n>>1;j<n_2;j+=rstep16, ++m_k)
	//	for(int j=0;j<n;j+=rstep*8, ++m_k)
	//	for(int j=0;j<m_2;j+=8, ++m_k)
		{
			p.m_r[m_k]=_mm256_setr_epi16(roots[j], roots[j+rstep], roots[j+rstep*2], roots[j+rstep*3], roots[j+rstep*4], roots[j+rstep*5], roots[j+rstep*6], roots[j+rstep*7],
				roots[j+rstep*8], roots[j+rstep*9], roots[j+rstep*10], roots[j+rstep*11], roots[j+rstep*12], roots[j+rstep*13], roots[j+rstep*14], roots[j+rstep*15]);
			__m256i v_lo=_mm256_mullo_epi16(p.m_r[m_k], m_beta2);
			__m256i v_hi=_mm256_mulhi_epi16(p.m_r[m_k], m_beta2);
			v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
			v_lo=_mm256_mulhi_epi16(v_lo, m_q);
			v_lo=_mm256_sub_epi16(v_hi, v_lo);
			__m256i cmp_mask=_mm256_cmpgt_epi16(m_zero, v_lo);
			cmp_mask=_mm256_and_si256(cmp_mask, m_q);
			p.m_r[m_k]=_mm256_add_epi16(v_lo, cmp_mask);
		//std::cout<<m_k<<'\t', print_register(p.m_r[m_k], q, -1);//
				
			p.m_ir[m_k]=_mm256_setr_epi16(iroots[j], iroots[j+rstep], iroots[j+rstep*2], iroots[j+rstep*3], iroots[j+rstep*4], iroots[j+rstep*5], iroots[j+rstep*6], iroots[j+rstep*7],
				iroots[j+rstep*8], iroots[j+rstep*9], iroots[j+rstep*10], iroots[j+rstep*11], iroots[j+rstep*12], iroots[j+rstep*13], iroots[j+rstep*14], iroots[j+rstep*15]);
			v_lo=_mm256_mullo_epi16(p.m_ir[m_k], m_beta2);
			v_hi=_mm256_mulhi_epi16(p.m_ir[m_k], m_beta2);
			v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
			v_lo=_mm256_mulhi_epi16(v_lo, m_q);
			v_lo=_mm256_sub_epi16(v_hi, v_lo);
			cmp_mask=_mm256_cmpgt_epi16(m_zero, v_lo);
			cmp_mask=_mm256_and_si256(cmp_mask, m_q);
			p.m_ir[m_k]=_mm256_add_epi16(v_lo, cmp_mask);
		//std::cout<<m_k<<'\t', print_register(p.m_ir[m_k], q, -1);//

			//std::cout<<j<<"\t";
			//for(int k3=0;k3<16;++k3)
			//	printf(" %5d", p.m_r[m_k].m256i_i16[k3]*2304%q);
			//std::cout<<endl;
		}
	}

	p.m_stage=(__m256i*)_aligned_malloc(10*sizeof(__m256i), sizeof(__m256i));//stg 2, 3, 4, 5, 6_1, 6_2, 7_1, 7_2, 7_3, 7_4
	p.m_istage=(__m256i*)_aligned_malloc(10*sizeof(__m256i), sizeof(__m256i));

	p.m_stage[0]=_mm256_set1_epi32(roots[n>>2]<<16|1);
	p.m_stage[1]=_mm256_setr_epi16(1, 1, 1, roots[n>>3], 1, roots[n>>2], 1, roots[n*3>>3], 1, 1, 1, roots[n>>3], 1, roots[n>>2], 1, roots[n*3>>3]);

	p.m_stage[2]=_mm256_setr_epi16(	1, roots[n>>4], roots[n>>3], roots[n*3>>4], roots[n>>2], roots[n*5>>4], roots[n*3>>3], roots[n*7>>4],
									1, roots[n>>4], roots[n>>3], roots[n*3>>4], roots[n>>2], roots[n*5>>4], roots[n*3>>3], roots[n*7>>4]);
	
	p.m_stage[3]=_mm256_setr_epi16(1, roots[n>>5], roots[n>>4], roots[n*3>>5], roots[n>>3], roots[n*5>>5], roots[n*6>>5], roots[n*7>>5],
					roots[n*8>>5], roots[n*9>>5], roots[n*10>>5], roots[n*11>>5], roots[n*12>>5], roots[n*13>>5], roots[n*14>>5], roots[n*15>>5]);
	
	p.m_stage[4]=_mm256_setr_epi16(					1, roots[n>>6], roots[n>>5], roots[n*3>>6], roots[n>>4], roots[n*5>>6], roots[n*6>>6], roots[n*7>>6],
									roots[n*8>>6], roots[n*9>>6], roots[n*10>>6], roots[n*11>>6], roots[n*12>>6], roots[n*13>>6], roots[n*14>>6], roots[n*15>>6]);
	p.m_stage[5]=_mm256_setr_epi16(	roots[n*16>>6], roots[n*17>>6], roots[n*18>>6], roots[n*19>>6], roots[n*20>>6], roots[n*21>>6], roots[n*22>>6], roots[n*23>>6],
									roots[n*24>>6], roots[n*25>>6], roots[n*26>>6], roots[n*27>>6], roots[n*28>>6], roots[n*29>>6], roots[n*30>>6], roots[n*31>>6]);
	
	p.m_stage[6]=_mm256_setr_epi16(					1, roots[n>>7], roots[n>>6], roots[n*3>>7], roots[n>>5], roots[n*5>>7], roots[n*6>>7], roots[n*7>>7],
									roots[n*8>>7], roots[n*9>>7], roots[n*10>>7], roots[n*11>>7], roots[n*12>>7], roots[n*13>>7], roots[n*14>>7], roots[n*15>>7]);
	p.m_stage[7]=_mm256_setr_epi16(	roots[n*16>>7], roots[n*17>>7], roots[n*18>>7], roots[n*19>>7], roots[n*20>>7], roots[n*21>>7], roots[n*22>>7], roots[n*23>>7],
									roots[n*24>>7], roots[n*25>>7], roots[n*26>>7], roots[n*27>>7], roots[n*28>>7], roots[n*29>>7], roots[n*30>>7], roots[n*31>>7]);
	p.m_stage[8]=_mm256_setr_epi16(	roots[n*32>>7], roots[n*33>>7], roots[n*34>>7], roots[n*35>>7], roots[n*36>>7], roots[n*37>>7], roots[n*38>>7], roots[n*39>>7],
									roots[n*40>>7], roots[n*41>>7], roots[n*42>>7], roots[n*43>>7], roots[n*44>>7], roots[n*45>>7], roots[n*46>>7], roots[n*47>>7]);
	p.m_stage[9]=_mm256_setr_epi16(	roots[n*48>>7], roots[n*49>>7], roots[n*50>>7], roots[n*51>>7], roots[n*52>>7], roots[n*53>>7], roots[n*54>>7], roots[n*55>>7],
									roots[n*56>>7], roots[n*57>>7], roots[n*58>>7], roots[n*59>>7], roots[n*60>>7], roots[n*61>>7], roots[n*62>>7], roots[n*63>>7]);
		
	p.m_istage[0]=_mm256_set1_epi32((iroots[n>>2])<<16|1);//stage 2
	p.m_istage[1]=_mm256_setr_epi16(1, 1, 1, iroots[n>>3], 1, iroots[n>>2], 1, iroots[n*3>>3], 1, 1, 1, iroots[n>>3], 1, iroots[n>>2], 1, iroots[n*3>>3]);//stage 3

	p.m_istage[2]=_mm256_setr_epi16(	1, iroots[n>>4], iroots[n>>3], iroots[n*3>>4], iroots[n>>2], iroots[n*5>>4], iroots[n*3>>3], iroots[n*7>>4],//stage 4
										1, iroots[n>>4], iroots[n>>3], iroots[n*3>>4], iroots[n>>2], iroots[n*5>>4], iroots[n*3>>3], iroots[n*7>>4]);
	
	p.m_istage[3]=_mm256_setr_epi16(	1, iroots[n>>5], iroots[n>>4], iroots[n*3>>5], iroots[n>>3], iroots[n*5>>5], iroots[n*6>>5], iroots[n*7>>5],//stage 5
						iroots[n*8>>5], iroots[n*9>>5], iroots[n*10>>5], iroots[n*11>>5], iroots[n*12>>5], iroots[n*13>>5], iroots[n*14>>5], iroots[n*15>>5]);
	
	p.m_istage[4]=_mm256_setr_epi16(			1, iroots[n>>6], iroots[n>>5], iroots[n*3>>6], iroots[n>>4], iroots[n*5>>6], iroots[n*6>>6], iroots[n*7>>6],//stage 6
									iroots[n*8>>6], iroots[n*9>>6], iroots[n*10>>6], iroots[n*11>>6], iroots[n*12>>6], iroots[n*13>>6], iroots[n*14>>6], iroots[n*15>>6]);
	p.m_istage[5]=_mm256_setr_epi16(iroots[n*16>>6], iroots[n*17>>6], iroots[n*18>>6], iroots[n*19>>6], iroots[n*20>>6], iroots[n*21>>6], iroots[n*22>>6], iroots[n*23>>6],
									iroots[n*24>>6], iroots[n*25>>6], iroots[n*26>>6], iroots[n*27>>6], iroots[n*28>>6], iroots[n*29>>6], iroots[n*30>>6], iroots[n*31>>6]);
	
	p.m_istage[6]=_mm256_setr_epi16(			1, iroots[n>>7], iroots[n>>6], iroots[n*3>>7], iroots[n>>5], iroots[n*5>>7], iroots[n*6>>7], iroots[n*7>>7],//stage 7
									iroots[n*8>>7], iroots[n*9>>7], iroots[n*10>>7], iroots[n*11>>7], iroots[n*12>>7], iroots[n*13>>7], iroots[n*14>>7], iroots[n*15>>7]);
	p.m_istage[7]=_mm256_setr_epi16(iroots[n*16>>7], iroots[n*17>>7], iroots[n*18>>7], iroots[n*19>>7], iroots[n*20>>7], iroots[n*21>>7], iroots[n*22>>7], iroots[n*23>>7],
									iroots[n*24>>7], iroots[n*25>>7], iroots[n*26>>7], iroots[n*27>>7], iroots[n*28>>7], iroots[n*29>>7], iroots[n*30>>7], iroots[n*31>>7]);
	p.m_istage[8]=_mm256_setr_epi16(iroots[n*32>>7], iroots[n*33>>7], iroots[n*34>>7], iroots[n*35>>7], iroots[n*36>>7], iroots[n*37>>7], iroots[n*38>>7], iroots[n*39>>7],
									iroots[n*40>>7], iroots[n*41>>7], iroots[n*42>>7], iroots[n*43>>7], iroots[n*44>>7], iroots[n*45>>7], iroots[n*46>>7], iroots[n*47>>7]);
	p.m_istage[9]=_mm256_setr_epi16(iroots[n*48>>7], iroots[n*49>>7], iroots[n*50>>7], iroots[n*51>>7], iroots[n*52>>7], iroots[n*53>>7], iroots[n*54>>7], iroots[n*55>>7],
									iroots[n*56>>7], iroots[n*57>>7], iroots[n*58>>7], iroots[n*59>>7], iroots[n*60>>7], iroots[n*61>>7], iroots[n*62>>7], iroots[n*63>>7]);

	for(int k=0;k<10;++k)
	{
		__m256i v_lo=_mm256_mullo_epi16(p.m_stage[k], m_beta2);
		__m256i v_hi=_mm256_mulhi_epi16(p.m_stage[k], m_beta2);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		v_lo=_mm256_sub_epi16(v_hi, v_lo);
		__m256i cmp_mask=_mm256_cmpgt_epi16(m_zero, v_lo);
		cmp_mask=_mm256_and_si256(cmp_mask, m_q);
		p.m_stage[k]=_mm256_add_epi16(v_lo, cmp_mask);

		v_lo=_mm256_mullo_epi16(p.m_istage[k], m_beta2);
		v_hi=_mm256_mulhi_epi16(p.m_istage[k], m_beta2);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		v_lo=_mm256_sub_epi16(v_hi, v_lo);
		cmp_mask=_mm256_cmpgt_epi16(m_zero, v_lo);
		cmp_mask=_mm256_and_si256(cmp_mask, m_q);
		p.m_istage[k]=_mm256_add_epi16(v_lo, cmp_mask);
		//print_register(p.m_stage[k], q);//
		//print_register(p.m_istage[k], q);//
	}
	
	_aligned_free(roots), _aligned_free(iroots);
	_m_empty();
}
void		number_transform_destroy_avx2(NTT_params_AVX2 &p)
{
	_aligned_free(p.m_phi), _aligned_free(p.m_iphi);
	_aligned_free(p.m_r), _aligned_free(p.m_ir);
	_aligned_free(p.m_stage), _aligned_free(p.m_istage);
}
void		number_transform_avx2(short const *src, short *Dst, int q, int q_1, int n, short sbar_m, const __m256i *m_r, const __m256i *m_stage)
{
	short const *a=src;
	short *A=Dst;
	//for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
	//	A[k]=a[bitreverse_table[k]];
	
	const __m256i bar_m=_mm256_set1_epi16(sbar_m);//short Barrett reduction
	const __m256i m_zero=_mm256_setzero_si256();
	const __m256i m_q=_mm256_set1_epi16(q);
	const __m256i m_q_1=_mm256_set1_epi16(q_1);

	for(int k=0;k<n;k+=128)//first stage
	{
		__m256i va1=_mm256_load_si256((__m256i*)(A+k));
		__m256i va2=_mm256_load_si256((__m256i*)(A+k+16));
		__m256i va3=_mm256_load_si256((__m256i*)(A+k+32));
		__m256i va4=_mm256_load_si256((__m256i*)(A+k+48));
		__m256i va5=_mm256_load_si256((__m256i*)(A+k+64));
		__m256i va6=_mm256_load_si256((__m256i*)(A+k+80));
		__m256i va7=_mm256_load_si256((__m256i*)(A+k+96));
		__m256i va8=_mm256_load_si256((__m256i*)(A+k+112));
	//if(!k)print_register("input:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//

		__m256i sum1=_mm256_hadd_epi16(va1, va2);//stage 1
		__m256i dif1=_mm256_hsub_epi16(va1, va2);
		__m256i sum2=_mm256_hadd_epi16(va3, va4);
		__m256i dif2=_mm256_hsub_epi16(va3, va4);
		__m256i sum3=_mm256_hadd_epi16(va5, va6);
		__m256i dif3=_mm256_hsub_epi16(va5, va6);
		__m256i sum4=_mm256_hadd_epi16(va7, va8);
		__m256i dif4=_mm256_hsub_epi16(va7, va8);
		sum1=_mm256_permute4x64_epi64(sum1, _MM_SHUFFLE(3, 1, 2, 0));
		dif1=_mm256_permute4x64_epi64(dif1, _MM_SHUFFLE(3, 1, 2, 0));
		sum2=_mm256_permute4x64_epi64(sum2, _MM_SHUFFLE(3, 1, 2, 0));
		dif2=_mm256_permute4x64_epi64(dif2, _MM_SHUFFLE(3, 1, 2, 0));
		sum3=_mm256_permute4x64_epi64(sum3, _MM_SHUFFLE(3, 1, 2, 0));
		dif3=_mm256_permute4x64_epi64(dif3, _MM_SHUFFLE(3, 1, 2, 0));
		sum4=_mm256_permute4x64_epi64(sum4, _MM_SHUFFLE(3, 1, 2, 0));
		dif4=_mm256_permute4x64_epi64(dif4, _MM_SHUFFLE(3, 1, 2, 0));
	//if(!k)print_register("stage 1:\n", sum1, q), print_register(dif1, q), print_register(sum2, q), print_register(dif2, q);//
		
		__m256i temp=_mm256_mulhi_epi16(sum1, bar_m);//short Barrett reduction
		temp=_mm256_mullo_epi16(temp, m_q);
		sum1=_mm256_sub_epi16(sum1, temp);
		temp=_mm256_mulhi_epi16(sum2, bar_m);
		temp=_mm256_mullo_epi16(temp, m_q);
		sum2=_mm256_sub_epi16(sum2, temp);
		temp=_mm256_mulhi_epi16(sum3, bar_m);
		temp=_mm256_mullo_epi16(temp, m_q);
		sum3=_mm256_sub_epi16(sum3, temp);
		temp=_mm256_mulhi_epi16(sum4, bar_m);
		temp=_mm256_mullo_epi16(temp, m_q);
		sum4=_mm256_sub_epi16(sum4, temp);
		
		__m256i v_lo=_mm256_mullo_epi16(dif1, m_stage[0]);//stage 2
		__m256i v_hi=_mm256_mulhi_epi16(dif1, m_stage[0]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		dif1=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(dif2, m_stage[0]);
		v_hi=_mm256_mulhi_epi16(dif2, m_stage[0]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		dif2=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(dif3, m_stage[0]);
		v_hi=_mm256_mulhi_epi16(dif3, m_stage[0]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		dif3=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(dif4, m_stage[0]);
		v_hi=_mm256_mulhi_epi16(dif4, m_stage[0]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		dif4=_mm256_sub_epi16(v_hi, v_lo);
	//if(!k)print_register("*wn4:\n", sum1, q), print_register(dif1, q), print_register(sum2, q), print_register(dif2, q);//
		
		__m256i t0=_mm256_hadd_epi16(sum1, dif1);
		__m256i t1=_mm256_hsub_epi16(sum1, dif1);
		va1=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));
		va2=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(3, 1, 3, 1)));
		t0=_mm256_hadd_epi16(sum2, dif2);
		t1=_mm256_hsub_epi16(sum2, dif2);
		va3=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));
		va4=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(3, 1, 3, 1)));
		t0=_mm256_hadd_epi16(sum3, dif3);
		t1=_mm256_hsub_epi16(sum3, dif3);
		va5=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));
		va6=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(3, 1, 3, 1)));
		t0=_mm256_hadd_epi16(sum4, dif4);
		t1=_mm256_hsub_epi16(sum4, dif4);
		va7=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));
		va8=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(t0), _mm256_castsi256_ps(t1), _MM_SHUFFLE(3, 1, 3, 1)));
	//if(!k)print_register("stage 2:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
	
	//if(!k)print_register("m_stage[1]:\n", m_stage[1], q);
		v_lo=_mm256_mullo_epi16(va1, m_stage[1]);//stage 3
		v_hi=_mm256_mulhi_epi16(va1, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled by m_stage
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va1=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va2, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va2, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va2=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va3, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va3, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va3=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va4, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va4, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va4=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va5, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va5, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va5=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va6, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va6, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va6=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va7, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va7, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va7=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va8, m_stage[1]);
		v_hi=_mm256_mulhi_epi16(va8, m_stage[1]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va8=_mm256_sub_epi16(v_hi, v_lo);
	//if(!k)print_register("*wn8:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
		sum1=_mm256_hadd_epi16(va1, va2);
		dif1=_mm256_hsub_epi16(va1, va2);
		va1=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum1), _mm256_castsi256_ps(dif1), _MM_SHUFFLE(1, 0, 1, 0)));
		va2=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum1), _mm256_castsi256_ps(dif1), _MM_SHUFFLE(3, 2, 3, 2)));
		sum2=_mm256_hadd_epi16(va3, va4);
		dif2=_mm256_hsub_epi16(va3, va4);
		va3=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum2), _mm256_castsi256_ps(dif2), _MM_SHUFFLE(1, 0, 1, 0)));
		va4=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum2), _mm256_castsi256_ps(dif2), _MM_SHUFFLE(3, 2, 3, 2)));
		sum3=_mm256_hadd_epi16(va5, va6);
		dif3=_mm256_hsub_epi16(va5, va6);
		va5=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum3), _mm256_castsi256_ps(dif3), _MM_SHUFFLE(1, 0, 1, 0)));
		va6=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum3), _mm256_castsi256_ps(dif3), _MM_SHUFFLE(3, 2, 3, 2)));
		sum4=_mm256_hadd_epi16(va7, va8);
		dif4=_mm256_hsub_epi16(va7, va8);
		va7=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum4), _mm256_castsi256_ps(dif4), _MM_SHUFFLE(1, 0, 1, 0)));
		va8=_mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(sum4), _mm256_castsi256_ps(dif4), _MM_SHUFFLE(3, 2, 3, 2)));
	//if(!k)print_register("stage 3:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
		v_lo=_mm256_mullo_epi16(va2, m_stage[2]);//stage 4
		v_hi=_mm256_mulhi_epi16(va2, m_stage[2]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va2=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va4, m_stage[2]);
		v_hi=_mm256_mulhi_epi16(va4, m_stage[2]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va4=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va6, m_stage[2]);
		v_hi=_mm256_mulhi_epi16(va6, m_stage[2]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va6=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va8, m_stage[2]);
		v_hi=_mm256_mulhi_epi16(va8, m_stage[2]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va8=_mm256_sub_epi16(v_hi, v_lo);
		
		temp=_mm256_mulhi_epi16(va1, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va1=_mm256_sub_epi16(va1, temp);//short Barrett reduction
		temp=_mm256_mulhi_epi16(va3, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va3=_mm256_sub_epi16(va3, temp);
		temp=_mm256_mulhi_epi16(va5, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va5=_mm256_sub_epi16(va5, temp);
		temp=_mm256_mulhi_epi16(va7, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va7=_mm256_sub_epi16(va7, temp);
	//if(!k)print_register("*wn16:\n", va2, q), print_register(va4, q);//

		sum1=_mm256_add_epi16(va1, va2);
		dif1=_mm256_sub_epi16(va1, va2);
		sum2=_mm256_add_epi16(va3, va4);
		dif2=_mm256_sub_epi16(va3, va4);
		sum3=_mm256_add_epi16(va5, va6);
		dif3=_mm256_sub_epi16(va5, va6);
		sum4=_mm256_add_epi16(va7, va8);
		dif4=_mm256_sub_epi16(va7, va8);
		va1=_mm256_permute2f128_si256(sum1, dif1, 2<<4|0);
		va2=_mm256_permute2f128_si256(sum1, dif1, 3<<4|1);
		va3=_mm256_permute2f128_si256(sum2, dif2, 2<<4|0);
		va4=_mm256_permute2f128_si256(sum2, dif2, 3<<4|1);
		va5=_mm256_permute2f128_si256(sum3, dif3, 2<<4|0);
		va6=_mm256_permute2f128_si256(sum3, dif3, 3<<4|1);
		va7=_mm256_permute2f128_si256(sum4, dif4, 2<<4|0);
		va8=_mm256_permute2f128_si256(sum4, dif4, 3<<4|1);
	//if(!k)print_register("stage 4:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		//_mm256_store_si256((__m256i*)(A+k), va1);
		//_mm256_store_si256((__m256i*)(A+k+16), va2);
		//_mm256_store_si256((__m256i*)(A+k+32), va3);
		//_mm256_store_si256((__m256i*)(A+k+48), va4);
		//_mm256_store_si256((__m256i*)(A+k+64), va5);
		//_mm256_store_si256((__m256i*)(A+k+80), va6);
		//_mm256_store_si256((__m256i*)(A+k+96), va7);
		//_mm256_store_si256((__m256i*)(A+k+112), va8);

		v_lo=_mm256_mullo_epi16(va2, m_stage[3]);//stage 5
		v_hi=_mm256_mulhi_epi16(va2, m_stage[3]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va2=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va4, m_stage[3]);
		v_hi=_mm256_mulhi_epi16(va4, m_stage[3]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va4=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va6, m_stage[3]);
		v_hi=_mm256_mulhi_epi16(va6, m_stage[3]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va6=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va8, m_stage[3]);
		v_hi=_mm256_mulhi_epi16(va8, m_stage[3]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va8=_mm256_sub_epi16(v_hi, v_lo);

		temp=_mm256_mulhi_epi16(va1, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va1=_mm256_sub_epi16(va1, temp);//short Barrett reduction
		temp=_mm256_mulhi_epi16(va3, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va3=_mm256_sub_epi16(va3, temp);
		temp=_mm256_mulhi_epi16(va5, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va5=_mm256_sub_epi16(va5, temp);
		temp=_mm256_mulhi_epi16(va7, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va7=_mm256_sub_epi16(va7, temp);
	//if(!k)print_register("*wn32:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//

		__m256i vb1=_mm256_add_epi16(va1, va2);
		__m256i vb2=_mm256_sub_epi16(va1, va2);
		__m256i vb3=_mm256_add_epi16(va3, va4);
		__m256i vb4=_mm256_sub_epi16(va3, va4);
		__m256i vb5=_mm256_add_epi16(va5, va6);
		__m256i vb6=_mm256_sub_epi16(va5, va6);
		__m256i vb7=_mm256_add_epi16(va7, va8);
		__m256i vb8=_mm256_sub_epi16(va7, va8);
	//if(!k)print_register("stage 5:\n", vb1, q), print_register(vb2, q), print_register(vb3, q), print_register(vb4, q);//
		//_mm256_store_si256((__m256i*)(A+k), sum1);
		//_mm256_store_si256((__m256i*)(A+k+16), dif1);
		//_mm256_store_si256((__m256i*)(A+k+32), sum2);
		//_mm256_store_si256((__m256i*)(A+k+48), dif2);
		//_mm256_store_si256((__m256i*)(A+k+64), sum3);
		//_mm256_store_si256((__m256i*)(A+k+80), dif3);
		//_mm256_store_si256((__m256i*)(A+k+96), sum4);
		//_mm256_store_si256((__m256i*)(A+k+112), dif4);
		
		v_lo=_mm256_mullo_epi16(vb3, m_stage[4]);//stage 6
		v_hi=_mm256_mulhi_epi16(vb3, m_stage[4]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		vb3=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(vb4, m_stage[5]);
		v_hi=_mm256_mulhi_epi16(vb4, m_stage[5]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		vb4=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(vb7, m_stage[4]);
		v_hi=_mm256_mulhi_epi16(vb7, m_stage[4]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		vb7=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(vb8, m_stage[5]);
		v_hi=_mm256_mulhi_epi16(vb8, m_stage[5]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		vb8=_mm256_sub_epi16(v_hi, v_lo);

		temp=_mm256_mulhi_epi16(vb1, bar_m), temp=_mm256_mullo_epi16(temp, m_q), vb1=_mm256_sub_epi16(vb1, temp);//short Barrett reduction
		temp=_mm256_mulhi_epi16(vb2, bar_m), temp=_mm256_mullo_epi16(temp, m_q), vb2=_mm256_sub_epi16(vb2, temp);
		temp=_mm256_mulhi_epi16(vb5, bar_m), temp=_mm256_mullo_epi16(temp, m_q), vb5=_mm256_sub_epi16(vb5, temp);
		temp=_mm256_mulhi_epi16(vb6, bar_m), temp=_mm256_mullo_epi16(temp, m_q), vb6=_mm256_sub_epi16(vb6, temp);
	//if(!k)print_register("*wn64:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
		va1=_mm256_add_epi16(vb1, vb3);
		va2=_mm256_add_epi16(vb2, vb4);
		va3=_mm256_sub_epi16(vb1, vb3);
		va4=_mm256_sub_epi16(vb2, vb4);
		va5=_mm256_add_epi16(vb5, vb7);
		va6=_mm256_add_epi16(vb6, vb8);
		va7=_mm256_sub_epi16(vb5, vb7);
		va8=_mm256_sub_epi16(vb6, vb8);
	//if(!k)print_register("stage 6:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q),		print_register(va5, q), print_register(va6, q), print_register(va7, q), print_register(va8, q);//
		//_mm256_store_si256((__m256i*)(A+k), va1);
		//_mm256_store_si256((__m256i*)(A+k+16), va2);
		//_mm256_store_si256((__m256i*)(A+k+32), va3);
		//_mm256_store_si256((__m256i*)(A+k+48), va4);
		//_mm256_store_si256((__m256i*)(A+k+64), va5);
		//_mm256_store_si256((__m256i*)(A+k+80), va6);
		//_mm256_store_si256((__m256i*)(A+k+96), va7);
		//_mm256_store_si256((__m256i*)(A+k+112), va8);

		v_lo=_mm256_mullo_epi16(va5, m_stage[6]);//stage 7
		v_hi=_mm256_mulhi_epi16(va5, m_stage[6]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va5=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va6, m_stage[7]);
		v_hi=_mm256_mulhi_epi16(va6, m_stage[7]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va6=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va7, m_stage[8]);
		v_hi=_mm256_mulhi_epi16(va7, m_stage[8]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va7=_mm256_sub_epi16(v_hi, v_lo);
		v_lo=_mm256_mullo_epi16(va8, m_stage[9]);
		v_hi=_mm256_mulhi_epi16(va8, m_stage[9]);
		v_lo=_mm256_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm256_mulhi_epi16(v_lo, m_q);
		va8=_mm256_sub_epi16(v_hi, v_lo);

		temp=_mm256_mulhi_epi16(va1, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va1=_mm256_sub_epi16(va1, temp);//short Barrett reduction
		temp=_mm256_mulhi_epi16(va2, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va2=_mm256_sub_epi16(va2, temp);
		temp=_mm256_mulhi_epi16(va3, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va3=_mm256_sub_epi16(va3, temp);
		temp=_mm256_mulhi_epi16(va4, bar_m), temp=_mm256_mullo_epi16(temp, m_q), va4=_mm256_sub_epi16(va4, temp);
	//if(!k)print_register("m_stage[6~9]:\n", m_stage[6], q), print_register(m_stage[7], q), print_register(m_stage[8], q), print_register(m_stage[9], q);
	//if(!k)print_register("*wn128:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q),		print_register(va5, q), print_register(va6, q), print_register(va7, q), print_register(va8, q);//

		vb1=_mm256_add_epi16(va1, va5);
		vb2=_mm256_add_epi16(va2, va6);
		vb3=_mm256_add_epi16(va3, va7);
		vb4=_mm256_add_epi16(va4, va8);
		vb5=_mm256_sub_epi16(va1, va5);
		vb6=_mm256_sub_epi16(va2, va6);
		vb7=_mm256_sub_epi16(va3, va7);
		vb8=_mm256_sub_epi16(va4, va8);
		_mm256_store_si256((__m256i*)(A+k), vb1);
		_mm256_store_si256((__m256i*)(A+k+16), vb2);
		_mm256_store_si256((__m256i*)(A+k+32), vb3);
		_mm256_store_si256((__m256i*)(A+k+48), vb4);
		_mm256_store_si256((__m256i*)(A+k+64), vb5);
		_mm256_store_si256((__m256i*)(A+k+80), vb6);
		_mm256_store_si256((__m256i*)(A+k+96), vb7);
		_mm256_store_si256((__m256i*)(A+k+112), vb8);
	//if(!k)print_register("stage 7 store:\n", vb1, q), print_register(vb2, q), print_register(vb3, q), print_register(vb4, q),	print_register(vb5, q), print_register(vb6, q), print_register(vb7, q), print_register(vb8, q);//
	}//*/
	
	for(int m=256;m<=n;m<<=1)//stage
	{
		int nblocks=n/m, nblocks8=nblocks<<3, m_2=m>>1;
		for(int k=0;k<n;k+=m)//block
		{
			for(int j=0, kr=(m>>5)-8;j<m_2;j+=64, kr+=4)//operation		//kr equation!
		//	for(int j=0, kr=(m>>5)-1;j<m_2;j+=64, kr+=4)
			{
				__m256i v16_1=_mm256_load_si256((__m256i*)(A+k+j));
				__m256i v16_2=_mm256_load_si256((__m256i*)(A+k+j+16));
				__m256i v16_3=_mm256_load_si256((__m256i*)(A+k+j+32));
				__m256i v16_4=_mm256_load_si256((__m256i*)(A+k+j+48));
				__m256i v16_5=_mm256_load_si256((__m256i*)(A+k+j+m_2));
				__m256i v16_6=_mm256_load_si256((__m256i*)(A+k+j+m_2+16));
				__m256i v16_7=_mm256_load_si256((__m256i*)(A+k+j+m_2+32));
				__m256i v16_8=_mm256_load_si256((__m256i*)(A+k+j+m_2+48));
			//if(m==256&&!k&&!j)//
			//{
			//	print_register("\nstage 8 input:\n", v16_1, q), print_register(v16_2, q), print_register(v16_3, q), print_register(v16_4, q),	print_register(v16_5, q), print_register(v16_6, q), print_register(v16_7, q), print_register(v16_8, q);//
			//	print_register("m_r:\n", m_r[kr], q), print_register(m_r[kr+1], q), print_register(m_r[kr+2], q), print_register(m_r[kr+3], q);//
			//}

				//montgomery reduction, cancelled by m_r
				__m256i vb_lo=_mm256_mullo_epi16(v16_5, m_r[kr]), vb_hi=_mm256_mulhi_epi16(v16_5, m_r[kr]);	vb_lo=_mm256_mullo_epi16(vb_lo, m_q_1), vb_lo=_mm256_mulhi_epi16(vb_lo, m_q), v16_5=_mm256_sub_epi16(vb_hi, vb_lo);//montgomery reduction, cancelled by m_r
				vb_lo=_mm256_mullo_epi16(v16_6, m_r[kr+1]), vb_hi=_mm256_mulhi_epi16(v16_6, m_r[kr+1]), vb_lo=_mm256_mullo_epi16(vb_lo, m_q_1), vb_lo=_mm256_mulhi_epi16(vb_lo, m_q), v16_6=_mm256_sub_epi16(vb_hi, vb_lo);
				vb_lo=_mm256_mullo_epi16(v16_7, m_r[kr+2]), vb_hi=_mm256_mulhi_epi16(v16_7, m_r[kr+2]), vb_lo=_mm256_mullo_epi16(vb_lo, m_q_1), vb_lo=_mm256_mulhi_epi16(vb_lo, m_q), v16_7=_mm256_sub_epi16(vb_hi, vb_lo);
				vb_lo=_mm256_mullo_epi16(v16_8, m_r[kr+3]), vb_hi=_mm256_mulhi_epi16(v16_8, m_r[kr+3]), vb_lo=_mm256_mullo_epi16(vb_lo, m_q_1), vb_lo=_mm256_mulhi_epi16(vb_lo, m_q), v16_8=_mm256_sub_epi16(vb_hi, vb_lo);
			//if(m==256&&!k&&!j)print_register("in * m_r:\n", v16_5, q), print_register(v16_6, q), print_register(v16_7, q), print_register(v16_8, q);//

				__m256i sum1=_mm256_add_epi16(v16_1, v16_5);
				__m256i sum2=_mm256_add_epi16(v16_2, v16_6);
				__m256i sum3=_mm256_add_epi16(v16_3, v16_7);
				__m256i sum4=_mm256_add_epi16(v16_4, v16_8);
				__m256i dif1=_mm256_sub_epi16(v16_1, v16_5);
				__m256i dif2=_mm256_sub_epi16(v16_2, v16_6);
				__m256i dif3=_mm256_sub_epi16(v16_3, v16_7);
				__m256i dif4=_mm256_sub_epi16(v16_4, v16_8);

				__m256i cmp_mask=_mm256_cmpgt_epi16(m_zero, sum1);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum1=_mm256_add_epi16(sum1, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, sum2);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum2=_mm256_add_epi16(sum2, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, sum3);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum3=_mm256_add_epi16(sum3, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, sum4);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum4=_mm256_add_epi16(sum4, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, dif1);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif1=_mm256_add_epi16(dif1, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, dif2);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif2=_mm256_add_epi16(dif2, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, dif3);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif3=_mm256_add_epi16(dif3, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(m_zero, dif4);			cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif4=_mm256_add_epi16(dif4, cmp_mask);

				cmp_mask=_mm256_cmpgt_epi16(sum1, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum1=_mm256_sub_epi16(sum1, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(sum2, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum2=_mm256_sub_epi16(sum2, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(sum3, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum3=_mm256_sub_epi16(sum3, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(sum4, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	sum4=_mm256_sub_epi16(sum4, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(dif1, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif1=_mm256_sub_epi16(dif1, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(dif2, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif2=_mm256_sub_epi16(dif2, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(dif3, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif3=_mm256_sub_epi16(dif3, cmp_mask);
				cmp_mask=_mm256_cmpgt_epi16(dif4, m_q);	cmp_mask=_mm256_and_si256(cmp_mask, m_q);	dif4=_mm256_sub_epi16(dif4, cmp_mask);
				
			//if(m==256&&!k&&!j)print_register("\nstage 8 store:\n", sum1, q), print_register(sum2, q), print_register(sum3, q), print_register(sum4, q),	print_register(dif1, q), print_register(dif2, q), print_register(dif3, q), print_register(dif4, q);//
				_mm256_store_si256((__m256i*)(A+k+j), sum1);
				_mm256_store_si256((__m256i*)(A+k+j+16), sum2);
				_mm256_store_si256((__m256i*)(A+k+j+32), sum3);
				_mm256_store_si256((__m256i*)(A+k+j+48), sum4);
				_mm256_store_si256((__m256i*)(A+k+j+m_2), dif1);
				_mm256_store_si256((__m256i*)(A+k+j+m_2+16), dif2);
				_mm256_store_si256((__m256i*)(A+k+j+m_2+32), dif3);
				_mm256_store_si256((__m256i*)(A+k+j+m_2+48), dif4);
			}
		}
		//std::cout<<"stg "<<log_2(m)<<':', print_element(A, n, q);//
	}
	_m_empty();
}
void		apply_NTT_avx2(short *src, short *Dst, NTT_params_AVX2 const &p, bool forward_BRP)
{
	//std::cout<<"input\t", print_element(src, p.n, p.q);//
	short n=p.n, q=p.q, q_1=p.q_1;
	if(forward_BRP)
		for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
			Dst[k]=src[bitreverse_table[k]];
	else
		memcpy(Dst, src, n*sizeof(short));
//	std::cout<<"a\t", print_element(src, n);//
	if(p.anti_cyclic)
	{
		__m256i m_q=_mm256_set1_epi16(q);
		__m256i m_q_1=_mm256_set1_epi16(q_1);
		__m256i m_zero=_mm256_setzero_si256();

		int n_16=n>>4;
		for(int k=0;k<n_16;++k)
		{
			//print_register(p.m_phi[k], p.q);//
			__m256i va=_mm256_load_si256((__m256i*)Dst+k);
			__m256i va_lo=_mm256_mullo_epi16(va, p.m_phi[k]);
			__m256i va_hi=_mm256_mulhi_epi16(va, p.m_phi[k]);
			va_lo=_mm256_mullo_epi16(va_lo, m_q_1);//montgomery reduction
			va_lo=_mm256_mulhi_epi16(va_lo, m_q);
			va_lo=_mm256_sub_epi16(va_hi, va_lo);

			__m256i cmp_mask=_mm256_cmpgt_epi16(m_zero, va_lo);
			cmp_mask=_mm256_and_si256(cmp_mask, m_q);
			va_lo=_mm256_add_epi16(va_lo, cmp_mask);
			_mm256_store_si256((__m256i*)Dst+k, va_lo);//no BRP
		//	_mm256_store_si256((__m256i*)src+k, va_lo);
		}
		_m_empty();
	}
	//std::cout<<"BRP[a * phi]\t", print_element(Dst, p.n, p.q);//
	number_transform_avx2(src, Dst, q, q_1, n, p.sbar_m, p.m_r, p.m_stage);
}
void		apply_inverse_NTT_avx2(short const *src, short *Dst, NTT_params_AVX2 const &p)
{
	short n=p.n, q=p.q, q_1=p.q_1;
	//std::cout<<"NTT\t", print_element(src, n, q);//
	for(int k=0;k<n;++k)//bit reverse permutation
		Dst[k]=src[bitreverse_table[k]];
	number_transform_avx2(src, Dst, q, q_1, n, p.sbar_m, p.m_ir, p.m_istage);
	//std::cout<<"INTT before multiplication:", print_element(Dst, n, q);//

	__m256i m_q=_mm256_set1_epi16(q);
	__m256i m_q_1=_mm256_set1_epi16(q_1);
	__m256i m_zero=_mm256_setzero_si256();
	if(p.anti_cyclic)
	{
		int n_16=n>>4;
		for(int k=0;k<n_16;++k)
		{
			__m256i va=_mm256_load_si256((__m256i*)Dst+k);
			__m256i va_lo=_mm256_mullo_epi16(va, p.m_iphi[k]);
			__m256i va_hi=_mm256_mulhi_epi16(va, p.m_iphi[k]);
			va_lo=_mm256_mullo_epi16(va_lo, m_q_1);//montgomery reduction
			va_lo=_mm256_mulhi_epi16(va_lo, m_q);
			va_lo=_mm256_sub_epi16(va_hi, va_lo);

			__m256i cmp_mask=_mm256_cmpgt_epi16(m_zero, va_lo);
			cmp_mask=_mm256_and_si256(cmp_mask, m_q);
			va_lo=_mm256_add_epi16(va_lo, cmp_mask);
			_mm256_store_si256((__m256i*)Dst+k, va_lo);
		}
	}
	else
	{
		__m256i m_q=_mm256_set1_epi16(q);
		__m256i m_q_1=_mm256_set1_epi16(q_1);
		__m256i m_zero=_mm256_setzero_si256();
		__m256i m_n_1=_mm256_set1_epi16(p.n_1*p.beta_q%q);
		for(int k=0;k<n;k+=16)
		{
			__m256i va=_mm256_load_si256((__m256i*)(Dst+k));
			__m256i va_lo=_mm256_mullo_epi16(va, m_n_1);
			__m256i va_hi=_mm256_mulhi_epi16(va, m_n_1);

			va_lo=_mm256_mullo_epi16(va_lo, m_q_1);//montgomery reduction
			va_lo=_mm256_mulhi_epi16(va_lo, m_q);
			va_lo=_mm256_sub_epi16(va_hi, va_lo);
			
			//__m256i cmp_mask=_mm256_cmplt_epi16(m_q, va_lo);
			//cmp_mask=_mm256_and_si128(cmp_mask, m_q);
			//va_lo=_mm256_sub_epi16(va_lo, cmp_mask);
			__m256i cmp_mask=_mm256_cmpgt_epi16(m_zero, va_lo);
			cmp_mask=_mm256_and_si256(cmp_mask, m_q);
			va_lo=_mm256_add_epi16(va_lo, cmp_mask);
			_mm256_store_si256((__m256i*)(Dst+k), va_lo);
		}
	}
	//std::cout<<"*n_1:", print_element(Dst, n, q);//
	_m_empty();
}
#endif
//struct NTT_params_SSE
//{
//	short n_1, q_1;
//	__m128i *m_phi, *m_iphi, *m_r, *m_ir, *m_stage, *m_istage;
//};
void		number_transform_initialize_sse(short n, short q, short w, short sqrt_w, bool anti_cyclic, NTT_params_SSE &p)
{
	p.n=n, p.q=q, p.w=w, p.sqrt_w=sqrt_w, p.sbar_m=0x10000/q, p.anti_cyclic=anti_cyclic;
	number_transform_initialize_begin(n, q, p.n_1, p.beta_q, p.q_1);

	bitreverse_init(n, log_2(n));
	short *roots=(short*)_aligned_malloc(n*sizeof(short), sizeof(__m128i)), *iroots=(short*)_aligned_malloc(n*sizeof(short), sizeof(__m128i));
//	short *roots=(short*)malloc(n*sizeof(short)), *iroots=(short*)malloc(n*sizeof(short));
	for(int k=0, wk=1;k<n;++k)//initialize NTT roots
		roots[k]=wk, iroots[(2*n-k)%n]=wk, wk*=w, wk%=q;
	
	int n_8=n>>3;
	p.m_phi=(__m128i*)_aligned_malloc(n_8*sizeof(__m128i), sizeof(__m128i));//powers of phi=sqrt(w)
	p.m_iphi=(__m128i*)_aligned_malloc(n_8*sizeof(__m128i), sizeof(__m128i));

	const __m128i m_q=_mm_set1_epi16(q);
	const __m128i m_q_1=_mm_set1_epi16(p.q_1);
	const __m128i m_zero=_mm_setzero_si128();
	const __m128i m_beta2=_mm_set1_epi16(p.beta_q*p.beta_q%q);//r2	10952*		1: 4091, 2: 10952, 3: 11227, 4: 5664 (stg4), 5: 6659, 6: 9545 (6), 7: 6442, 8: 6606 (8), 9: 1635, 10: 3569 (10), 11: 1447, 12: 8668, 13: 7023, 14: 11700, 15: 11334
	{
		short *t_phi=(short*)malloc(n*sizeof(short));
#ifdef IA32_USE_MONTGOMERY_REDUCTION
		int factor=p.beta_q;
		int f2=p.sqrt_w*p.beta_q%q;
#else
		int factor=p.beta_q;//%
#endif
		for(int k=0;k<n;++k)//{1, phi, ..., phi^(n-1)}	//218ms
		{
			t_phi[k]=factor;
#ifdef IA32_USE_MONTGOMERY_REDUCTION
			int pr=factor*f2;	factor=short(*(short*)&pr*p.q_1), factor=factor*q>>16, factor=((short*)&pr)[1]-factor;//, factor+=q&-(factor<0);
#else
			factor=factor*p.sqrt_w%q, factor+=q&-(factor<0);
#endif
		}
		auto p_phi=(short*)p.m_phi;
		//std::cout<<"phi:", print_element(t_phi, n, q, -1);//
		for(int k=0;k<n;++k)
			p_phi[k]=t_phi[bitreverse_table[k]];
		//std::cout<<"BRP[phi]:", print_element(p_phi, n, q, -1);//
		free(t_phi);//*/
	/*	__m128i m_factor=_mm_set1_epi16(roots[4]*p.beta_q%q);
		__m128i m_wpowers=_mm_setr_epi16(1, sqrt_w, roots[1], roots[1]*sqrt_w%q,	roots[2], roots[2]*sqrt_w%q, roots[3], roots[3]*sqrt_w%q);
		__m128i m_powers_lo=_mm_mullo_epi16(m_wpowers, m_beta2);
		m_wpowers=_mm_mulhi_epi16(m_wpowers, m_beta2);
		m_powers_lo=_mm_mullo_epi16(m_powers_lo, m_q_1);
		m_powers_lo=_mm_mulhi_epi16(m_powers_lo, m_q);
		m_wpowers=_mm_sub_epi16(m_wpowers, m_powers_lo);
		_mm_store_si128(p.m_phi, m_wpowers);
		for(int k=1;k<n_8;++k)
		{
			__m128i m_powers_lo=_mm_mullo_epi16(m_wpowers, m_factor);
			m_wpowers=_mm_mulhi_epi16(m_wpowers, m_factor);
			m_powers_lo=_mm_mullo_epi16(m_powers_lo, m_q_1);//montgomery reduction
			m_powers_lo=_mm_mulhi_epi16(m_powers_lo, m_q);
			m_wpowers=_mm_sub_epi16(m_wpowers, m_powers_lo);
			_mm_store_si128(p.m_phi+k, m_wpowers);
		}//*/

		__m128i m_factor=_mm_set1_epi16(iroots[n-4]*p.beta_q%q);
		__m128i m_wpowers=_mm_setr_epi16(
			(q-iroots[n-4])*p.n_1%q, (q-iroots[n-3]*sqrt_w%q)*p.n_1%q, (q-iroots[n-3])*p.n_1%q, (q-iroots[n-2]*sqrt_w%q)*p.n_1%q,
			(q-iroots[n-2])*p.n_1%q, (q-iroots[n-1]*sqrt_w%q)*p.n_1%q, (q-iroots[n-1])*p.n_1%q, (q-sqrt_w)*p.n_1%q);
		__m128i m_powers_lo=_mm_mullo_epi16(m_wpowers, m_beta2);
		m_wpowers=_mm_mulhi_epi16(m_wpowers, m_beta2);
		m_powers_lo=_mm_mullo_epi16(m_powers_lo, m_q_1);
		m_powers_lo=_mm_mulhi_epi16(m_powers_lo, m_q);
		m_wpowers=_mm_sub_epi16(m_wpowers, m_powers_lo);
		_mm_store_si128(p.m_iphi+n_8-1, m_wpowers);
		for(int k=n_8-2;k>=0;--k)
		{
			__m128i m_powers_lo=_mm_mullo_epi16(m_wpowers, m_factor);
			m_wpowers=_mm_mulhi_epi16(m_wpowers, m_factor);
			m_powers_lo=_mm_mullo_epi16(m_powers_lo, m_q_1);
			m_powers_lo=_mm_mulhi_epi16(m_powers_lo, m_q);
			m_wpowers=_mm_sub_epi16(m_wpowers, m_powers_lo);
			_mm_store_si128(p.m_iphi+k, m_wpowers);
			//print_register(p.m_iphi[k], q);//
		}
	}
	p.m_r=(__m128i*)_aligned_malloc(n_8*sizeof(__m128i), sizeof(__m128i));
	p.m_ir=(__m128i*)_aligned_malloc(n_8*sizeof(__m128i), sizeof(__m128i));
	//{
	//	const __m128i m_q=_mm_set1_epi16(q);
	//	const __m128i m_q_1=_mm_set1_epi16(p.q_1);
	//	const __m128i m_zero=_mm_setzero_si128();
	//	const __m128i m_beta2=_mm_set1_epi16(beta_q*beta_q%q);
		for(int rstep=n>>6, rstep8=rstep<<3, m_k=0;rstep>0;rstep>>=1, rstep8>>=1)
	//	for(int rstep=n/32, rstep8=rstep<<3, m_k=0;rstep>0;rstep>>=1, rstep8>>=1)
	//	for(int rstep=32, rstep8=rstep<<3, m_k=0;rstep>0;rstep>>=1, rstep8>>=1)
		//	for(int m_k=0, m_2=16, rstep=32;m_2<128;m_2<<=1, rstep>>=2)
		//	for(int m_k=0, m_2=16, rstep=32;m_2<1024;m_2<<=1, rstep>>=2)
		{
			for(int j=0, n_2=n>>1;j<n_2;j+=rstep8, ++m_k)
		//	for(int j=0;j<n;j+=rstep*8, ++m_k)
		//	for(int j=0;j<m_2;j+=8, ++m_k)
			{
				p.m_r[m_k]=_mm_setr_epi16(roots[j], roots[j+rstep], roots[j+rstep*2], roots[j+rstep*3], roots[j+rstep*4], roots[j+rstep*5], roots[j+rstep*6], roots[j+rstep*7]);
				__m128i v_lo=_mm_mullo_epi16(p.m_r[m_k], m_beta2);
				__m128i v_hi=_mm_mulhi_epi16(p.m_r[m_k], m_beta2);
				v_lo=_mm_mullo_epi16(v_lo, m_q_1);
				v_lo=_mm_mulhi_epi16(v_lo, m_q);
				v_lo=_mm_sub_epi16(v_hi, v_lo);
				__m128i cmp_mask=_mm_cmplt_epi16(v_lo, m_zero);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				p.m_r[m_k]=_mm_add_epi16(v_lo, cmp_mask);
			//std::cout<<m_k<<'\t', print_register(p.m_r[m_k], q, -1);//
					
				p.m_ir[m_k]=_mm_setr_epi16(iroots[j], iroots[j+rstep], iroots[j+rstep*2], iroots[j+rstep*3], iroots[j+rstep*4], iroots[j+rstep*5], iroots[j+rstep*6], iroots[j+rstep*7]);
				v_lo=_mm_mullo_epi16(p.m_ir[m_k], m_beta2);
				v_hi=_mm_mulhi_epi16(p.m_ir[m_k], m_beta2);
				v_lo=_mm_mullo_epi16(v_lo, m_q_1);
				v_lo=_mm_mulhi_epi16(v_lo, m_q);
				v_lo=_mm_sub_epi16(v_hi, v_lo);
				cmp_mask=_mm_cmplt_epi16(v_lo, m_zero);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				p.m_ir[m_k]=_mm_add_epi16(v_lo, cmp_mask);
			//std::cout<<m_k<<'\t', print_register(p.m_ir[m_k], q, -1);//

				//std::cout<<j<<"\t";
				//for(int k3=0;k3<8;++k3)
				//	printf(" %5d", p.m_r[m_k].m128i_i16[k3]*2304%q);
				//std::cout<<endl;
			}
		}
	//}
	p.m_stage=(__m128i*)_aligned_malloc(5*sizeof(__m128i), sizeof(__m128i));//stg 2, 3, 4, 5_1, 5_2
	p.m_istage=(__m128i*)_aligned_malloc(5*sizeof(__m128i), sizeof(__m128i));

	p.m_stage[0]=_mm_set1_epi32(roots[n>>2]<<16|1);//stage 2
	p.m_stage[1]=_mm_setr_epi16(1, 1, 1, roots[n>>3], 1, roots[n>>2], 1, roots[n*3>>3]);//stage 3
	p.m_stage[2]=_mm_setr_epi16(1, roots[n>>4], roots[n>>3], roots[n*3>>4], roots[n>>2], roots[n*5>>4], roots[n*3>>3], roots[n*7>>4]);//stage 4
	p.m_stage[3]=_mm_setr_epi16(1, roots[n>>5], roots[n>>4], roots[n*3>>5], roots[n>>3], roots[n*5>>5], roots[n*6>>5], roots[n*7>>5]);						//stage 5_1
	p.m_stage[4]=_mm_setr_epi16(roots[n>>2], roots[n*9>>5], roots[n*10>>5], roots[n*11>>5], roots[n*12>>5], roots[n*13>>5], roots[n*14>>5], roots[n*15>>5]);//stage 5_2

	p.m_istage[0]=_mm_set1_epi32(iroots[n>>2]<<16|1);
	p.m_istage[1]=_mm_setr_epi16(1, 1, 1, iroots[n>>3], 1, iroots[n>>2], 1, iroots[n*3>>3]);
	p.m_istage[2]=_mm_setr_epi16(1, iroots[n>>4], iroots[n>>3], iroots[n*3>>4], iroots[n>>2], iroots[n*5>>4], iroots[n*3>>3], iroots[n*7>>4]);
	p.m_istage[3]=_mm_setr_epi16(1, iroots[n>>5], iroots[n>>4], iroots[n*3>>5], iroots[n>>3], iroots[n*5>>5], iroots[n*6>>5], iroots[n*7>>5]);
	p.m_istage[4]=_mm_setr_epi16(iroots[n>>2], iroots[n*9>>5], iroots[n*10>>5], iroots[n*11>>5], iroots[n*12>>5], iroots[n*13>>5], iroots[n*14>>5], iroots[n*15>>5]);

	for(int k=0;k<5;++k)
	{
		__m128i v_lo=_mm_mullo_epi16(p.m_stage[k], m_beta2);
		__m128i v_hi=_mm_mulhi_epi16(p.m_stage[k], m_beta2);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		v_lo=_mm_sub_epi16(v_hi, v_lo);
		__m128i cmp_mask=_mm_cmplt_epi16(v_lo, m_zero);
		cmp_mask=_mm_and_si128(cmp_mask, m_q);
		p.m_stage[k]=_mm_add_epi16(v_lo, cmp_mask);

		v_lo=_mm_mullo_epi16(p.m_istage[k], m_beta2);
		v_hi=_mm_mulhi_epi16(p.m_istage[k], m_beta2);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		v_lo=_mm_sub_epi16(v_hi, v_lo);
		cmp_mask=_mm_cmplt_epi16(v_lo, m_zero);
		cmp_mask=_mm_and_si128(cmp_mask, m_q);
		p.m_istage[k]=_mm_add_epi16(v_lo, cmp_mask);
		//print_register(p.m_stage[k], q);//
		//print_register(p.m_istage[k], q);//
	}

//	//stage 2
//	p.m_stage[0]=_mm_set1_epi32(roots[n>>2]*beta_q%q<<16|beta_q);
//
//	//stage 3
//	p.m_stage[1]=_mm_setr_epi16(beta_q, beta_q, beta_q, roots[n>>3]*beta_q%q, beta_q, roots[n>>2]*beta_q%q, beta_q, roots[n*3>>3]*beta_q%q);
////	p.m_stage[1]=_mm_setr_epi16(1, 1, 1, roots[n>>3], 1, roots[n>>2], 1, roots[n*3>>3]);
//
//	//stage 4
//	p.m_stage[2]=_mm_setr_epi16(beta_q, roots[n>>4]*beta_q%q, roots[n>>3]*beta_q%q, roots[n*3>>4]*beta_q%q, roots[n>>2]*beta_q%q, roots[n*5>>4]*beta_q%q, roots[n*3>>3]*beta_q%q, roots[n*7>>4]*beta_q%q);
//	
//	//stage 5
//	p.m_stage[3]=_mm_setr_epi16(beta_q, roots[n>>5]*beta_q%q, roots[n>>4]*beta_q%q, roots[n*3>>5]*beta_q%q, roots[n>>3]*beta_q%q, roots[n*5>>5]*beta_q%q, roots[n*6>>5]*beta_q%q, roots[n*7>>5]*beta_q%q);
//	p.m_stage[4]=_mm_setr_epi16(roots[n>>2]*beta_q%q, roots[n*9>>5]*beta_q%q, roots[n*10>>5]*beta_q%q, roots[n*11>>5]*beta_q%q, roots[n*12>>5]*beta_q%q, roots[n*13>>5]*beta_q%q, roots[n*14>>5]*beta_q%q, roots[n*15>>5]*beta_q%q);
//
//	p.m_istage[0]=_mm_set1_epi32(iroots[n>>2]*beta_q%q<<16|beta_q);
//	p.m_istage[1]=_mm_setr_epi16(beta_q, beta_q, beta_q, iroots[n>>3]*beta_q%q, beta_q, iroots[n>>2]*beta_q%q, beta_q, iroots[n*3>>3]*beta_q%q);
////	p.m_istage[1]=_mm_setr_epi16(1, 1, 1, iroots[n>>3], 1, iroots[n>>2], 1, iroots[n*3>>3]);
//
//	p.m_istage[2]=_mm_setr_epi16(beta_q, iroots[n>>4]*beta_q%q, iroots[n>>3]*beta_q%q, iroots[n*3>>4]*beta_q%q, iroots[n>>2]*beta_q%q, iroots[n*5>>4]*beta_q%q, iroots[n*3>>3]*beta_q%q, iroots[n*7>>4]*beta_q%q);
//		
//	p.m_istage[3]=_mm_setr_epi16(beta_q, iroots[n>>5]*beta_q%q, iroots[n>>4]*beta_q%q, iroots[n*3>>5]*beta_q%q, iroots[n>>3]*beta_q%q, iroots[n*5>>5]*beta_q%q, iroots[n*6>>5]*beta_q%q, iroots[n*7>>5]*beta_q%q);
//	p.m_istage[4]=_mm_setr_epi16(iroots[n>>2]*beta_q%q, iroots[n*9>>5]*beta_q%q, iroots[n*10>>5]*beta_q%q, iroots[n*11>>5]*beta_q%q, iroots[n*12>>5]*beta_q%q, iroots[n*13>>5]*beta_q%q, iroots[n*14>>5]*beta_q%q, iroots[n*15>>5]*beta_q%q);

	_aligned_free(roots), _aligned_free(iroots);
	_m_empty();
}
void		number_transform_destroy_sse(NTT_params_SSE &p)
{
	_aligned_free(p.m_phi), _aligned_free(p.m_iphi);
	_aligned_free(p.m_r), _aligned_free(p.m_ir);
	_aligned_free(p.m_stage), _aligned_free(p.m_istage);
}
void		number_transform_sse(short const *src, short *Dst, int q, int q_1, int n, short sbar_m, const __m128i *m_r, const __m128i *m_stage)
{
	short const *a=src;
	short *A=Dst;
	//for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
	//	A[k]=a[bitreverse_table[k]];
	
//std::cout<<"input:", print_element(A, n, q);//
	const __m128i bar_m=_mm_set1_epi16(sbar_m);//short Barrett reduction
	const __m128i m_zero=_mm_setzero_si128();
	const __m128i m_q=_mm_set1_epi16(q);//12289
	const __m128i m_q_1=_mm_set1_epi16(q_1);//-12287=53249
	
	for(int k=0;k<n;k+=32)//first stage
	{
		__m128i va1=_mm_load_si128((__m128i*)(A+k));	//23.4ms 51688c		//39.3ms 51013c
		__m128i va2=_mm_load_si128((__m128i*)(A+k+8));
		__m128i va3=_mm_load_si128((__m128i*)(A+k+16));
		__m128i va4=_mm_load_si128((__m128i*)(A+k+24));
	//if(!k)print_register("input:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
		__m128i sum1=_mm_hadd_epi16(va1, va2);//3 2	//stage 1
		__m128i dif1=_mm_hsub_epi16(va1, va2);//3 2
		__m128i sum2=_mm_hadd_epi16(va3, va4);//3 2
		__m128i dif2=_mm_hsub_epi16(va3, va4);//3 2
		
		__m128i temp=_mm_mulhi_epi16(sum1, bar_m);//5 1	//short Barrett reduction
		temp=_mm_mullo_epi16(temp, m_q);	//5 1
		sum1=_mm_sub_epi16(sum1, temp);		//1 0.5
		temp=_mm_mulhi_epi16(sum2, bar_m);	//5 1
		temp=_mm_mullo_epi16(temp, m_q);	//5 1
		sum2=_mm_sub_epi16(sum2, temp);		//1 0.5
	//if(!k)print_register("stage 1:\n", sum1, q), print_register(dif1, q), print_register(sum2, q), print_register(dif2, q);//
		
		__m128i v_lo=_mm_mullo_epi16(dif1, m_stage[0]);//5 1	//stage 2
		__m128i v_hi=_mm_mulhi_epi16(dif1, m_stage[0]);//5 1
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);		//5 1	//montgomery reduction, cancelled by m_stage[0]
		v_lo=_mm_mulhi_epi16(v_lo, m_q);		//5 1
		dif1=_mm_sub_epi16(v_hi, v_lo);			//1 0.5
		v_lo=_mm_mullo_epi16(dif2, m_stage[0]);	//5 1
		v_hi=_mm_mulhi_epi16(dif2, m_stage[0]);	//5 1
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);		//5 1
		v_lo=_mm_mulhi_epi16(v_lo, m_q);		//5 1
		dif2=_mm_sub_epi16(v_hi, v_lo);			//1 0.5
	//if(!k)print_register("*m_stage[0]:\n", sum1, q), print_register(dif1, q), print_register(sum1, q), print_register(dif1, q);//
		
		__m128i t0=_mm_hadd_epi16(sum1, dif1);	//3 2
		__m128i t1=_mm_hsub_epi16(sum1, dif1);	//3 2
		va1=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));//1 1
		va2=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(3, 1, 3, 1)));//1 1
		t0=_mm_hadd_epi16(sum2, dif2);			//3 2
		t1=_mm_hsub_epi16(sum2, dif2);			//3 2
		va3=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));//1 1
		va4=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(3, 1, 3, 1)));//1 1
	//if(!k)print_register("stage 2:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
	//if(!k)print_register("wn8:\n", m_stage[1], q);
		v_lo=_mm_mullo_epi16(va1, m_stage[1]);//stage 3
		v_hi=_mm_mulhi_epi16(va1, m_stage[1]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		va1=_mm_sub_epi16(v_hi, v_lo);
		v_lo=_mm_mullo_epi16(va2, m_stage[1]);
		v_hi=_mm_mulhi_epi16(va2, m_stage[1]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		va2=_mm_sub_epi16(v_hi, v_lo);
		v_lo=_mm_mullo_epi16(va3, m_stage[1]);
		v_hi=_mm_mulhi_epi16(va3, m_stage[1]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		va3=_mm_sub_epi16(v_hi, v_lo);
		v_lo=_mm_mullo_epi16(va4, m_stage[1]);
		v_hi=_mm_mulhi_epi16(va4, m_stage[1]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		va4=_mm_sub_epi16(v_hi, v_lo);
	//if(!k)print_register("*wn8:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
		sum1=_mm_hadd_epi16(va1, va2);
		dif1=_mm_hsub_epi16(va1, va2);
		sum2=_mm_hadd_epi16(va3, va4);
		dif2=_mm_hsub_epi16(va3, va4);
		va1=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(sum1), _mm_castsi128_ps(dif1), _MM_SHUFFLE(1, 0, 1, 0)));
		va2=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(sum1), _mm_castsi128_ps(dif1), _MM_SHUFFLE(3, 2, 3, 2)));
		va3=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(sum2), _mm_castsi128_ps(dif2), _MM_SHUFFLE(1, 0, 1, 0)));
		va4=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(sum2), _mm_castsi128_ps(dif2), _MM_SHUFFLE(3, 2, 3, 2)));
	//if(!k)print_register("stage 3:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//
		
		v_lo=_mm_mullo_epi16(va2, m_stage[2]);//stage 4
		v_hi=_mm_mulhi_epi16(va2, m_stage[2]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		va2=_mm_sub_epi16(v_hi, v_lo);
		v_lo=_mm_mullo_epi16(va4, m_stage[2]);
		v_hi=_mm_mulhi_epi16(va4, m_stage[2]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		va4=_mm_sub_epi16(v_hi, v_lo);
		
		temp=_mm_mulhi_epi16(va1, bar_m);//short Barrett reduction
		temp=_mm_mullo_epi16(temp, m_q);
		va1=_mm_sub_epi16(va1, temp);
		temp=_mm_mulhi_epi16(va3, bar_m);
		temp=_mm_mullo_epi16(temp, m_q);
		va3=_mm_sub_epi16(va3, temp);
		
		sum1=_mm_add_epi16(va1, va2);
		dif1=_mm_sub_epi16(va1, va2);
		sum2=_mm_add_epi16(va3, va4);
		dif2=_mm_sub_epi16(va3, va4);
	//if(!k)print_register("stage 4:\n", sum1, q), print_register(dif1, q), print_register(sum2, q), print_register(dif2, q);//
		
	//if(!k)print_register("stg 5 roots:\n", m_stage[3], q), print_register(m_stage[4], q);//
		v_lo=_mm_mullo_epi16(sum2, m_stage[3]);//stage 5
		v_hi=_mm_mulhi_epi16(sum2, m_stage[3]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);//montgomery reduction, cancelled
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		sum2=_mm_sub_epi16(v_hi, v_lo);
		v_lo=_mm_mullo_epi16(dif2, m_stage[4]);
		v_hi=_mm_mulhi_epi16(dif2, m_stage[4]);
		v_lo=_mm_mullo_epi16(v_lo, m_q_1);
		v_lo=_mm_mulhi_epi16(v_lo, m_q);
		dif2=_mm_sub_epi16(v_hi, v_lo);
		temp=_mm_mulhi_epi16(sum1, bar_m);//short Barrett reduction
		temp=_mm_mullo_epi16(temp, m_q);
		sum1=_mm_sub_epi16(sum1, temp);
		temp=_mm_mulhi_epi16(dif1, bar_m);
		temp=_mm_mullo_epi16(temp, m_q);
		dif1=_mm_sub_epi16(dif1, temp);
	//if(!k)print_register("*wn32:\n", sum1, q), print_register(dif1, q), print_register(sum2, q), print_register(dif2, q);//

		va1=_mm_add_epi16(sum1, sum2);
		va2=_mm_add_epi16(dif1, dif2);
		va3=_mm_sub_epi16(sum1, sum2);
		va4=_mm_sub_epi16(dif1, dif2);
		_mm_store_si128((__m128i*)(A+k), va1);
		_mm_store_si128((__m128i*)(A+k+8), va2);
		_mm_store_si128((__m128i*)(A+k+16), va3);
		_mm_store_si128((__m128i*)(A+k+24), va4);
	//if(!k)print_register("stage 5 store:\n", va1, q), print_register(va2, q), print_register(va3, q), print_register(va4, q);//*/
	}
	//std::cout<<"stg "<<log_2(32)<<':', print_element(A, n, q);//

	for(int m=64;m<=n;m<<=1)
	{
		int nblocks=n/m, nblocks8=nblocks<<3, m_2=m>>1;
		for(int k=0;k<n;k+=m)//block
		{
			for(int j=0, kr=(m>>4)-4;j<m_2;j+=16, kr+=2)//operation
		//	for(int j=0, kr=(m>>4)-2;j<m_2;j+=16, kr+=2)
			{
				__m128i v8_1=_mm_load_si128((__m128i*)(A+k+j));
				__m128i v8_2=_mm_load_si128((__m128i*)(A+k+j+8));
				__m128i v8_3=_mm_load_si128((__m128i*)(A+k+j+m_2));
				__m128i v8_4=_mm_load_si128((__m128i*)(A+k+j+m_2+8));
			//if(m==256&&!k&&j<=16)print_register("\nstage 8 input:\n", v8_1, q), print_register(v8_2, q), print_register(v8_3, q), print_register(v8_4, q);//

			//if(!k&&!j)print_register("m_r:\n", m_r[kr], q, -1), print_register(m_r[kr+1], q, -1);//
			//if(m==256&&!k&&j<=16)print_register("m_r:\n", m_r[kr], q), print_register(m_r[kr+1], q);//
				__m128i vb_lo=_mm_mullo_epi16(v8_3, m_r[kr]);
				__m128i vb_hi=_mm_mulhi_epi16(v8_3, m_r[kr]);
				vb_lo=_mm_mullo_epi16(vb_lo, m_q_1);//montgomery reduction, cancelled by m_r
				vb_lo=_mm_mulhi_epi16(vb_lo, m_q);
				v8_3=_mm_sub_epi16(vb_hi, vb_lo);
				vb_lo=_mm_mullo_epi16(v8_4, m_r[kr+1]);
				vb_hi=_mm_mulhi_epi16(v8_4, m_r[kr+1]);
				vb_lo=_mm_mullo_epi16(vb_lo, m_q_1);
				vb_lo=_mm_mulhi_epi16(vb_lo, m_q);
				v8_4=_mm_sub_epi16(vb_hi, vb_lo);

				__m128i sum1=_mm_add_epi16(v8_1, v8_3);
				__m128i sum2=_mm_add_epi16(v8_2, v8_4);
				__m128i dif1=_mm_sub_epi16(v8_1, v8_3);
				__m128i dif2=_mm_sub_epi16(v8_2, v8_4);
			//if(m==256&&!k&&j<=16)
			//{
			//	//print_register("stage 8 input:\n", v8_1, q), print_register(v8_2, q), print_register(v8_3, q), print_register(v8_4, q);//
			//	//print_register("m_r:\n", m_r[kr], q), print_register(m_r[kr+1], q);//
			//	print_register("v8_2*m_r:\n", v8_3, q), print_register(v8_4, q);//
			//	print_register("stage 8:\n", sum1, q), print_register(sum2, q), print_register(dif1, q), print_register(dif2, q);//
			//}
				
				__m128i cmp_mask=_mm_cmplt_epi16(sum1, m_zero);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				sum1=_mm_add_epi16(sum1, cmp_mask);
				cmp_mask=_mm_cmplt_epi16(sum2, m_zero);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				sum2=_mm_add_epi16(sum2, cmp_mask);
				cmp_mask=_mm_cmplt_epi16(dif1, m_zero);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				dif1=_mm_add_epi16(dif1, cmp_mask);
				cmp_mask=_mm_cmplt_epi16(dif2, m_zero);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				dif2=_mm_add_epi16(dif2, cmp_mask);

				cmp_mask=_mm_cmpgt_epi16(sum1, m_q);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				sum1=_mm_sub_epi16(sum1, cmp_mask);
				cmp_mask=_mm_cmpgt_epi16(sum2, m_q);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				sum2=_mm_sub_epi16(sum2, cmp_mask);
				cmp_mask=_mm_cmpgt_epi16(dif1, m_q);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				dif1=_mm_sub_epi16(dif1, cmp_mask);
				cmp_mask=_mm_cmpgt_epi16(dif2, m_q);
				cmp_mask=_mm_and_si128(cmp_mask, m_q);
				dif2=_mm_sub_epi16(dif2, cmp_mask);
			//if(m==128&&!k&&j<=16)print_register("stage 7 store:\n", sum1, q), print_register(sum2, q), print_register(dif1, q), print_register(dif2, q);//

				_mm_store_si128((__m128i*)(A+k+j), sum1);
				_mm_store_si128((__m128i*)(A+k+j+8), sum2);
				_mm_store_si128((__m128i*)(A+k+j+m_2), dif1);
				_mm_store_si128((__m128i*)(A+k+j+m_2+8), dif2);
			}
		}
		//std::cout<<"stg "<<log_2(m)<<':', print_element(A, n, q);//
	}
	_m_empty();
}
void		apply_NTT_sse(short *src, short *Dst, NTT_params_SSE const &p, bool forward_BRP)
//void		apply_NTT_modifies_src_sse(short *src, short *Dst, short q, short q_1, short n, short n_1, short sqrt_w, const __m128i *m_r, const __m128i *m_stage, const __m128i *m_phi, bool anti_cyclic)
{
	short n=p.n, q=p.q, q_1=p.q_1;
	//std::cout<<"\na:\t", print_element(src, n, q);//
	if(forward_BRP)
		for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
			Dst[k]=src[bitreverse_table[k]];
	else
		memcpy(Dst, src, n*sizeof(short));
	//std::cout<<"\nBRP:\t", print_element(Dst, n, q);//
	if(p.anti_cyclic)
	{
		__m128i m_q=_mm_set1_epi16(q);
		__m128i m_q_1=_mm_set1_epi16(q_1);
		__m128i m_zero=_mm_setzero_si128();

		int n_8=n>>3;
		//std::cout<<"\nm_phi:\n";//
		for(int k=0;k<n_8;++k)
		{
			//print_register(p.m_phi[k], p.q);//
			__m128i va=_mm_load_si128((__m128i*)Dst+k);
			__m128i va_lo=_mm_mullo_epi16(va, p.m_phi[k]);
			__m128i va_hi=_mm_mulhi_epi16(va, p.m_phi[k]);
			va_lo=_mm_mullo_epi16(va_lo, m_q_1);//montgomery reduction, cancelled by m_phi
			va_lo=_mm_mulhi_epi16(va_lo, m_q);
			va_lo=_mm_sub_epi16(va_hi, va_lo);

			__m128i cmp_mask=_mm_cmplt_epi16(va_lo, m_zero);
			cmp_mask=_mm_and_si128(cmp_mask, m_q);
			va_lo=_mm_add_epi16(va_lo, cmp_mask);
			_mm_store_si128((__m128i*)Dst+k, va_lo);
		}//*/
	}
	_m_empty();
	//std::cout<<"\nBRP[a * phi]:\t", print_element(Dst, n, q);//
	number_transform_sse(src, Dst, q, q_1, n, p.sbar_m, p.m_r, p.m_stage);
}
void		apply_inverse_NTT_sse(short const *src, short *Dst, NTT_params_SSE const &p)
{
	short n=p.n, q=p.q, q_1=p.q_1;
	//std::cout<<"NTT\t", print_element(src, n, q);//
	for(int k=0;k<n;++k)//bit reverse permutation
		Dst[k]=src[bitreverse_table[k]];
	number_transform_sse(src, Dst, q, q_1, n, p.sbar_m, p.m_ir, p.m_istage);
	//std::cout<<"INTT before multiplication:", print_element(Dst, n, q);//

	if(p.anti_cyclic)
	{
		__m128i m_q=_mm_set1_epi16(q);
		__m128i m_q_1=_mm_set1_epi16(q_1);
		__m128i m_zero=_mm_setzero_si128();

		int n_8=n>>3;
		for(int k=0;k<n_8;++k)
		{
			__m128i va=_mm_load_si128((__m128i*)Dst+k);
			__m128i va_lo=_mm_mullo_epi16(va, p.m_iphi[k]);
			__m128i va_hi=_mm_mulhi_epi16(va, p.m_iphi[k]);
			va_lo=_mm_mullo_epi16(va_lo, m_q_1);//montgomery reduction
			va_lo=_mm_mulhi_epi16(va_lo, m_q);
			va_lo=_mm_sub_epi16(va_hi, va_lo);

			__m128i cmp_mask=_mm_cmplt_epi16(va_lo, m_zero);
			cmp_mask=_mm_and_si128(cmp_mask, m_q);
			va_lo=_mm_add_epi16(va_lo, cmp_mask);
			_mm_store_si128((__m128i*)Dst+k, va_lo);
		}//*/
	}
	else
	{
		__m128i m_q=_mm_set1_epi16(q);
		__m128i m_q_1=_mm_set1_epi16(q_1);
		__m128i m_zero=_mm_setzero_si128();
		__m128i m_n_1=_mm_set1_epi16(p.n_1*p.beta_q%q);
		for(int k=0;k<n;k+=8)
		{
			__m128i va=_mm_load_si128((__m128i*)(Dst+k));
			__m128i va_lo=_mm_mullo_epi16(va, m_n_1);
			__m128i va_hi=_mm_mulhi_epi16(va, m_n_1);

			va_lo=_mm_mullo_epi16(va_lo, m_q_1);//montgomery reduction
			va_lo=_mm_mulhi_epi16(va_lo, m_q);
			va_lo=_mm_sub_epi16(va_hi, va_lo);
			
			//__m128i cmp_mask=_mm_cmplt_epi16(m_q, va_lo);
			//cmp_mask=_mm_and_si128(cmp_mask, m_q);
			//va_lo=_mm_sub_epi16(va_lo, cmp_mask);
			__m128i cmp_mask=_mm_cmplt_epi16(va_lo, m_zero);
			cmp_mask=_mm_and_si128(cmp_mask, m_q);
			va_lo=_mm_add_epi16(va_lo, cmp_mask);
			_mm_store_si128((__m128i*)(Dst+k), va_lo);
		}//*/
	}
	//std::cout<<"INTT*n_1:", print_element(Dst, n, q);//
	_m_empty();
}

//struct NTT_params_IA32
//{
//	short *roots, *iroots,
//		n_1, sqrt_w;
//};
void		number_transform_initialize_ia32(short n, short q, short w, short sqrt_w, bool anti_cyclic, NTT_params_IA32 &p)
{
	p.n=n, p.q=q, p.w=w, p.sqrt_w=sqrt_w, p.anti_cyclic=anti_cyclic;
	number_transform_initialize_begin(n, q, p.n_1, p.beta_q, p.q_1);

	int logn=log_2(n);
	p.beta_stg=calculate_montgomery_factor(q, logn);//beta^log_2(n), log_2(n)-1 stages + phi/correction
	//p.beta_stg=p.beta_q;
	//for(int k=1;k<logn;++k)
	//	p.beta_stg=p.beta_stg*p.beta_q%q;

	bitreverse_init(n, logn);

	p.roots=(short*)malloc(n*sizeof(short)), p.iroots=(short*)malloc(n*sizeof(short));
	for(int k=0, wk=1;k<n;++k)//initialize NTT roots
		p.roots[k]=wk, p.iroots[(2*n-k)%n]=wk, wk*=w, wk%=q;

	p.phi=(short*)malloc(n*sizeof(short));//bit-reverse permuted phi = BRP{1, sqrtw, ..., sqrtw^(n-1)}
	short *t_phi=(short*)malloc(n*sizeof(short));
#ifdef IA32_USE_MONTGOMERY_REDUCTION
	int factor=p.beta_stg;
	int f2=p.sqrt_w*p.beta_q%q;
#else
	int factor=1;//%
#endif
	for(int k=0;k<n;++k)//{1, phi, ..., phi^(n-1)}	//218ms
	{
		t_phi[k]=factor;
#ifdef IA32_USE_MONTGOMERY_REDUCTION
		int pr=factor*f2;	factor=short(*(short*)&pr*p.q_1), factor=factor*q>>16, factor=((short*)&pr)[1]-factor;//, factor+=q&-(factor<0);
#else
		factor=factor*p.sqrt_w%q, factor+=q&-(factor<0);
#endif
	}
	//std::cout<<"phi:", print_element(t_phi, n, q, -logn);//
	for(int k=0;k<n;++k)
		p.phi[k]=t_phi[bitreverse_table[k]];
	//std::cout<<"BRP[phi]:", print_element(p.phi, n, q, -logn);//
	free(t_phi);
}
void		number_transform_destroy_ia32(NTT_params_IA32 &p){free(p.roots), free(p.iroots), free(p.phi);}
void		number_transform_ia32(short const *src, short *Dst, short q, short q_1, short n, short const *roots)
{
	short const *a=src;
	short *A=Dst;
	//for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
	//	A[k]=a[bitreverse_table[k]];

/*	for(int m=n;m>=4;m>>=1)//stage
//	for(int m=4;m<=n;m*=2)
	{
		int rstep=n/m, m_2=m/2;
		for(int k=0;k<n;k+=m)//block
		{
			for(int j=0, kr=0;j<m_2;++j, kr+=rstep)//operation
			{
				auto &A0=A[k+j], &A1=A[k+j+m_2];
				
			//	int a0=A0+A1, a1=roots[kr]*(A0-A1);
			//	A0=a0;//128.1 166367

				int a0=A0+A1, a1=roots[kr]*(A0-A1);
				A0=short(*(short*)&a0*q_1)*q>>16, A0=((short*)&a0)[1]-A0;

				A1=short(*(short*)&a1*q_1)*q>>16, A1=((short*)&a1)[1]-A1;//U7700: 127.6 165770c
			}
		}
		m>>=1;
		if(m<4)
			break;
		rstep=n/m, m_2=m/2;
		for(int k=0;k<n;k+=m)//block
		{
			for(int j=0, kr=0;j<m_2;++j, kr+=rstep)//operation
			{
				auto &A0=A[k+j], &A1=A[k+j+m_2];
				
			//	int a0=A0+A1, a1=roots[kr]*(A0-A1);
			//	int b0=(a0*5>>16)*q; A0-=*(short*)&b0;//U7700: 129.4 168129c (121.6 157954c)

				int a0=A0+A1, a1=roots[kr]*(A0-A1);
				A0=short(*(short*)&a0*q_1)*q>>16, A0=((short*)&a0)[1]-A0;//montgomery reduction

				A1=short(*(short*)&a1*q_1)*q>>16, A1=((short*)&a1)[1]-A1;
			}
		}
	//	std::cout<<"stg "<<log_2(m)<<':', print_element(A, n);//
	}
	for(int k=0;k<n;k+=2)	//308 315 ms
	{
		auto &A0=A[k], &A1=A[k+1];
		auto a0=A0, a1=A1;
		A0=a0+a1, A1=a0-a1;				//308 312 ms
	//	A0=(a0+a1)%q, A1=(a0-a1)%q;		//327 328 330 ms
	//	A0+=q&-(A0<0), A1+=q&-(A1<0);	//331 332 336 ms
	//	A[k]=a0+a1, A[k+1]=a0-a1;//error=0
	}//*/

	//std::cout<<"input:", print_element(A, n, q, log_2(n));//
//	short A4[4]={A[0], A[1], A[2], A[3]};
	for(int k=0;k<n;k+=2)	//308 315 ms
	{
		auto &A0=A[k], &A1=A[k+1];
		auto a0=A0, a1=A1;
		A0=a0+a1, A1=a0-a1;				//308 312 ms
	//	A0=(a0+a1)%q, A1=(a0-a1)%q;		//327 328 330 ms
	//	A0+=q&-(A0<0), A1+=q&-(A1<0);	//331 332 336 ms
	//	A[k]=a0+a1, A[k+1]=a0-a1;//error=0
	}//*/
	//std::cout<<"stg "<<log_2(2)<<':', print_element(A, n, q, 1-log_2(n));//
//	int stg=2;
	for(int m=4;m<=n;m*=2)//stage
	{
		//if(m==32)
		//	int LOL_1=0;
		int rstep=n/m, m_2=m/2;
		for(int k=0;k<n;k+=m)//block
		{
			for(int j=0, kr=0;j<m_2;++j, kr+=rstep)//operation
			{
				auto &A0=A[k+j], &A1=A[k+j+m_2];
				int a0=A0, a1=roots[kr]*A1;
				//int sum=a0+a1;	short *pv=(short*)&sum;	pv[0]=(unsigned short)(pv[0]*53249)*12289>>16, A0=pv[1]-pv[0], A0+=q&-(A0<0);
				//int dif=a0-a1;		pv=(short*)&dif;	pv[0]=(unsigned short)(pv[0]*53249)*12289>>16, A1=pv[1]-pv[0], A1+=q&-(A1<0);
				////A0=A0*4091%q;
				////A1=A1*4091%q;
				//if(!k&&j<16)//
				//{
				//	std::cout<<m<<' '<<k<<' '<<j<<"\t\t";
				//		std::cout<<' '<<roots[kr];
				//	std::cout<<endl;
				//}
#ifdef IA32_USE_MONTGOMERY_REDUCTION
				int b0=a0+a1, b1=a0-a1;

				A0=short(*(short*)&b0*q_1);
				A0=A0*q>>16;
				A0=((short*)&b0)[1]-A0;//montgomery reduction
				A1=short(*(short*)&b1*q_1);
				A1=A1*q>>16;
				A1=((short*)&b1)[1]-A1;//U7700: 143.3 186090c

				//	A0=short(*(short*)&b0*q_1), A0=A0*q>>16, A0=((short*)&b0)[1]-A0, A0+=q&-(A0<0);//		226.02ms 499045c		//203ms 448239c
				//	A1=short(*(short*)&b1*q_1), A1=A1*q>>16, A1=((short*)&b1)[1]-A1, A1+=q&-(A1<0);//U7700: 195.4 253829c

					//A0=b0-(b0*10921LL>>27)*q, A0+=q&-(A0<0);//barrett reduction 418.63ms 924334c	//365.17ms 806297c
					//A1=b1-(b1*10921LL>>27)*q, A1+=q&-(A1<0);
#else
				A0=(a0+a1)%q, A0+=q&-(A0<0);//269.14ms 594262c		//242.32ms 535041c
				A1=(a0-a1)%q, A1+=q&-(A1<0);
#endif
			}
		}
	//	std::cout<<"stg "<<log_2(m)<<':', print_element(A, n, q);//%
		//	std::cout<<"stg "<<log_2(m)<<':', print_element(A, n, q, log_2(m)-log_2(n));//forward
		//	std::cout<<"stg "<<log_2(m)<<':', print_element(A, n, q, log_2(m)-1);//inverse
		//	std::cout<<"stg "<<log_2(m)<<':', print_element(A, n, q, 9-(log_2(m)-1));//
	//	++stg;
	}
}
void		apply_NTT_ia32(short *src, short *Dst, NTT_params_IA32 const &p, bool forward_BRP)
{
	short n=p.n, q=p.q, q_1=p.q_1;
	//std::cout<<"Input:", print_element(src, p.n, p.q);//

	if(forward_BRP)
		for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
			Dst[k]=src[bitreverse_table[k]];
	else
		memcpy(Dst, src, n*sizeof(short));
	//std::cout<<"BRP:", print_element(Dst, n, q);//
	if(p.anti_cyclic)
	{
		for(int k=0;k<n;++k)//*{1, phi, ..., phi^(n-1)}	//218ms
		{
			auto &vk=Dst[k];
#ifdef IA32_USE_MONTGOMERY_REDUCTION
			int pr=vk*p.phi[k];	vk=short(*(short*)&pr*q_1), vk=vk*q>>16, vk=((short*)&pr)[1]-vk;
#else
			vk=vk*p.phi[k]%q;
#endif
			vk+=q&-(vk<0);
		}
	}
#ifdef IA32_USE_MONTGOMERY_REDUCTION
	else
	{
	//	factor=calculate_montgomery_factor(p.q, log_2(n));
		for(int k=0;k<n;++k)
		{
			auto &vk=Dst[k];
			int pr=vk*p.beta_stg; vk=short(*(short*)&pr*q_1), vk=vk*q>>16, vk=((short*)&pr)[1]-vk;
		}
	}
	//std::cout<<"BRP:", print_element(Dst, n, q, 1-log_2(n));//
#endif

/*	if(anti_cyclic)//1: 4091, 2: 10952, 3: 11227, 4: 5664 (stg4), 5: 6659, 6: 9545 (6), 7: 6442, 8: 6606 (8), 9: 1635, 10: 3569 (10), 11: 1447, 12: 8668, 13: 7023, 14: 11700, 15: 11334
	{
#ifdef IA32_USE_MONTGOMERY_REDUCTION
		int factor=p.beta_stg;
		int f2=p.sqrt_w*p.beta_q%q;
#else
		int factor=1;//%
#endif
		for(int k=0;k<n;++k)//*{1, phi, ..., phi^(n-1)}	//218ms
		{
			auto &vk=src[k];
#ifdef IA32_USE_MONTGOMERY_REDUCTION
			int pr=vk*factor;	vk=short(*(short*)&pr*q_1), vk=vk*q>>16, vk=((short*)&pr)[1]-vk, vk+=q&-(vk<0);
			pr=factor*f2, factor=short(*(short*)&pr*q_1), factor=factor*q>>16, factor=((short*)&pr)[1]-factor;//, factor+=q&-(factor<0);
#else
			vk=vk*factor%q, vk+=q&-(vk<0);
			factor*=p.sqrt_w, factor%=q, factor+=q&-(factor<0);
#endif
		}
	}
	std::cout<<"input * phi:", print_element(src, n, q);//
	if(forward_BRP)
		for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
			Dst[k]=src[bitreverse_table[k]];//*/

//	std::cout<<"BRP[input * phi]:", print_element(src, n, q);//
/*	if(anti_cyclic)
	{
		int factor=1635;//r2 mon
	//	int factor=1;
		for(int k=0;k<n;++k)//*{1, phi, ..., phi^(n-1)}	//218ms
		{
			auto &vk=src[k];
			vk=vk*factor%q, vk+=q&-(vk<0);
		//	vk*=factor, vk%=q, vk+=q&-(vk<0);
			factor=factor*p.sqrt_w%q, factor+=q&-(factor<0);
		}
	}
	//print_element(src, n);//
	//else
	//	memcpy(Dst, src, n*sizeof(short));//*/
	number_transform_ia32(src, Dst, q, q_1, n, p.roots);
	//std::cout<<"NTT:", print_element(Dst, p.n, p.q);//
}
void		apply_inverse_NTT_ia32(short *src, short *Dst, NTT_params_IA32 const &p)
{
	//std::cout<<"NTT:", print_element(src, p.n, p.q);//
	short n=p.n, q=p.q, q_1=p.q_1;

	for(int k=0;k<n;++k)//bit reverse permutation	DIT NTT, DIF inverse-NTT
		Dst[k]=src[bitreverse_table[k]];
	number_transform_ia32(src, Dst, q, q_1, n, p.iroots);

//std::cout<<"before phi\t", print_element(Dst, n, q);//
#ifdef IA32_USE_MONTGOMERY_REDUCTION
	short correction=p.beta_stg;
#else
	short correction=1;//%
#endif
	short f2=p.sqrt_w*p.beta_q%q;
//	short f2=p.sqrt_w*4091%q;
	if(p.anti_cyclic)//1: 4091, 2: 10952, 3: 11227, 4: 5664 (stg4), 5: 6659, 6: 9545 (6), 7: 6442, 8: 6606 (8), 9: 1635, 10: 3569 (10), 11: 1447, 12: 8668, 13: 7023, 14: 11700, 15: 11334
	{
	//	p.sqrt_w=p.sqrt_w*4091%q;
		for(int k=n-1, factor=(q-p.sqrt_w)*p.n_1%q*correction%q;k>=0;--k)//*{1, -phi^(n-1), ..., -phi}
		{
			auto &vk=Dst[k];
#ifdef IA32_USE_MONTGOMERY_REDUCTION
			int pr=vk*factor;	vk=short(*(short*)&pr*q_1)*q>>16, vk=((short*)&pr)[1]-vk, vk+=q&-(vk<0);
			pr=factor*f2, factor=short(*(short*)&pr*q_1)*q>>16, factor=((short*)&pr)[1]-factor;//, factor+=q&-(factor<0);
#else
			vk=vk*factor%q, vk+=q&-(vk<0);
			factor*=p.sqrt_w, factor%=q, factor+=q&-(factor<0);
#endif
		}
	}
	else
	{
#ifdef IA32_USE_MONTGOMERY_REDUCTION
		short factor=p.beta_stg*p.n_1%q;
	//	short factor=calculate_montgomery_factor(q, log_2(n))*p.n_1%q;
		for(int k=0;k<n;++k)
		{
			auto &vk=Dst[k];
			int pr=vk*factor; vk=short(*(short*)&pr*q_1)*q>>16, vk=((short*)&pr)[1]-vk, vk+=q&-(vk<0);
		}
#else
	//	correction=n_1*4091%q*correction%q;
		for(int k=0;k<n;++k)
		{
			auto &vk=Dst[k];
		//	int pr=vk*correction;	vk=short(*(short*)&pr*q_1)*q>>16, vk=((short*)&pr)[1]-vk, vk+=q&-(vk<0);
			vk=vk*p.n_1%q;
		}
		//	Dst[k]=Dst[k]*n_1%q;
		//	Dst[k]*=n_1, Dst[k]%=q;//X
#endif
	}//*/

/*	//std::cout<<"/phi\t", print_element(Dst, n);//
//	int factor=6512;//mon x18 forward and inverse
	int factor=1635;//r2 mon x9
//	int factor=1;
	if(p.anti_cyclic)
	{
		factor=factor*p.n_1%q*(q-p.sqrt_w)%q;
		for(int k=n-1;k>=0;--k)//*{1, -phi^(n-1), ..., -phi}
		{
			auto &vk=Dst[k];
			vk=vk*factor%q, vk+=q&-(vk<0);
		//	vk*=-factor, vk%=q, vk+=q&-(vk<0);
			factor=factor*p.sqrt_w%q, factor+=q&-(factor<0);
		}
	}
	else
	{
		factor=factor*p.n_1%q;
		for(int k=0;k<n;++k)
			Dst[k]=Dst[k]*factor%q;
		//	Dst[k]*=n_1, Dst[k]%=q;//X
	}//*/
//std::cout<<"inverse NTT\t", print_element(Dst, p.n, p.q);//
}

//Strassen Algorithm		DEBUG
int			st_subtract=false;
void		matrix_multiplication_naive(int const *a, int const *b, int *c, int ah, int n, int bw, int q)
{
	for(int k=0;k<ah;++k)
	{
		for(int k2=0;k2<bw;++k2)
		{
			auto &ck=c[k*bw+k2];
		//	auto pa=a+k*n, pb=b+k2*bw;
		//	for(int k3=0;k3<n;++k3, ++pa, ++pb)
			for(int k3=0;k3<n;++k3)
				ck+=(a[k*n+k3]*b[k3*bw+k2]^-st_subtract)+st_subtract, ck%=q;//125.311 ms	125.957 ms	128.614 ms
			//	ck+=(*pa**pb^-st_subtract)+st_subtract, ck%=q;				//126.976 ms	133.685 ms
			//	ck+=(pa[k3]*pb[k3]^-st_subtract)+st_subtract, ck%=q;		//136.118 ms	slower!
			//	ck+=a[k*n+k3]*b[k3*bw+k2], ck%=q;
			ck+=q&-(ck<0);
		}
	}
}
void		matrix_add(int const *a, int const *b, int *c, int h, int w, int q, int negative, short a_inc, short b_inc, short c_inc)
{
	for(int k=0;k<h;++k, a+=a_inc, b+=b_inc, c+=c_inc)
		for(int k2=0;k2<w;++k2, ++a, ++b, ++c)
			*c=*a+(*b^-negative)+negative, *c%=q, *c+=q&-(*c<0);
}
void		matrix_add_special(int const *a1, int const *a2, int const *a3, int const *_a4, int *c, int h, int w, int q, short c_inc)
{
	for(int k=0;k<h;++k, c+=c_inc)
		for(int k2=0;k2<w;++k2, ++a1, ++a2, ++a3, ++_a4, ++c)
			*c=*a1+*a2+*a3-*_a4, *c%=q, *c+=q&-(*c<0);
}
void		copy_matrix(int const *a, int *c, int h, int w, short a_inc)
{
	for(int k=0;k<h;++k, a+=a_inc, c+=w)
		memcpy(c, a, w*sizeof(int));
}
void		matrix_multiplication_strassen(int const *a, int const *b, int *c, int *buffer, int ah, int n, int bw, int q)
{
//	if((n<=32)|!(ah%2)|!(bw%2))//141.817 ms	157.129 ms		152.686 ms
	if((n<=16)|!(ah%2)|!(bw%2))//127.524 ms	143.457 ms		138.603 ms	<-
//	if((n<= 8)|!(ah%2)|!(bw%2))//128.829 ms	161.634 ms		138.371 ms
//	if((n<= 4)|!(ah%2)|!(bw%2))//180.133 ms
//	if((n<= 2)|!(ah%2)|!(bw%2))//357.381 ms
		matrix_multiplication_naive(a, b, c, ah, n, bw, q);
	else
	{
		int const ah2=ah/2, n2=n/2, bw2=bw/2, size2=ah2*bw2;
		int tsize=ah2*n2, t2size=n2*bw2;
		int *M1=buffer, *M2=buffer+size2, *M3=buffer+size2*2, *M4=buffer+size2*3, *M5=buffer+size2*4, *M6=buffer+size2*5, *M7=buffer+size2*6,
			*t1=buffer+size2*7, *t2=t1+tsize, *b2=t2+t2size;
		memset(buffer, 0, size2*7*sizeof(int));
		int const
			*a11=a, *a12=a+n2 , *a21=a+ah2*n , *a22=a+ah2*n +n2 ,
			*b11=b, *b12=b+bw2, *b21=b+n2 *bw, *b22=b+n2 *bw+bw2;
		int *c11=c, *c12=c+bw2, *c21=c+ah2*bw, *c22=c+ah2*bw+bw2;
		
	//	std::vector<int> buf(buffer, buffer+size2*7+tsize+t2size);//
		matrix_add(a11, a22, t1, ah2, n2, q, 0, n2, n2, 0),	matrix_add(b11, b22, t2, n2, bw2, q, 0, bw2, bw2, 0),	matrix_multiplication_strassen(t1, t2, M1, b2, ah2, n2, bw2, q);//M1 = (A11+A22) (B11+B22)
		matrix_add(a21, a22, t1, ah2, n2, q, 0, n2, n2, 0),	copy_matrix(b11, t2, n2, bw2, bw),						matrix_multiplication_strassen(t1, t2, M2, b2, ah2, n2, bw2, q);//M2 = (A21+A22) B11
		copy_matrix(a11, t1, ah2, n2, n),					matrix_add(b12, b22, t2, n2, bw2, q, 1, bw2, bw2, 0),	matrix_multiplication_strassen(t1, t2, M3, b2, ah2, n2, bw2, q);//M3 = A11 (B12-B22)
		copy_matrix(a22, t1, ah2, n2, n),					matrix_add(b21, b11, t2, n2, bw2, q, 1, bw2, bw2, 0),	matrix_multiplication_strassen(t1, t2, M4, b2, ah2, n2, bw2, q);//M4 = A22 (B21-B11)
		matrix_add(a11, a12, t1, ah2, n2, q, 0, n2, n2, 0),	copy_matrix(b22, t2, n2, bw2, bw),						matrix_multiplication_strassen(t1, t2, M5, b2, ah2, n2, bw2, q);//M5 = (A11+A12) B22
		matrix_add(a21, a11, t1, ah2, n2, q, 1, n2, n2, 0),	matrix_add(b11, b12, t2, n2, bw2, q, 0, bw2, bw2, 0),	matrix_multiplication_strassen(t1, t2, M6, b2, ah2, n2, bw2, q);//M6 = (A21-A11) (B11+B12)
		matrix_add(a12, a22, t1, ah2, n2, q, 1, n2, n2, 0),	matrix_add(b21, b22, t2, n2, bw2, q, 0, bw2, bw2, 0),	matrix_multiplication_strassen(t1, t2, M7, b2, ah2, n2, bw2, q);//M7 = (A12-A22) (B21+B22)
	//	buf.assign(buffer, buffer+size2*7+tsize+t2size);//
		
	//	std::vector<int> c2(c, c+ah*bw);//
		matrix_add_special(M1, M4, M7, M5, c11, ah2, bw2, q, bw2);//C11 = M1+M4-M5+M7
		matrix_add(M3, M5, c12, ah2, bw2, q, 0, 0, 0, bw2);//C12 = M3+M5
		matrix_add(M2, M4, c21, ah2, bw2, q, 0, 0, 0, bw2);//C21 = M2+M4
		matrix_add_special(M1, M3, M6, M2, c22, ah2, bw2, q, bw2);//C22 = M1-M2+M3+M6
	//	c2.assign(c, c+ah*bw);//
	}
}
void		matrix_multiplication_strassen_add(int *C, int const *A, int const *B, int ah, int awbh, int bw, int q)
{
	int buffer_size=((ah*bw-4)*7+ah*awbh+awbh*bw)/3, *buffer=new int[buffer_size*2];//Mi ah*bw	t1 ah*awbh	t2 awbh*bw
	matrix_multiplication_strassen(A, B, C, buffer, ah, awbh, bw, q);
	memset(buffer, 0, buffer_size*sizeof(int));
	delete[] buffer;

/*	int *B2=new int[awbh*bw];
	for(int k2=0;k2<bw;++k2)
		for(int k=0;k<awbh;++k)
			B2[k2*awbh+k]=B[k*bw+k2];
	int buffer_size=((ah*bw-4)*7+ah*awbh+awbh*bw)/3, *buffer=new int[buffer_size*2];//Mi ah*bw	t1 ah*awbh	t2 awbh*bw
	matrix_multiplication_strassen(A, B2, C, buffer, ah, awbh, bw, q);
	memset(buffer, 0, buffer_size*sizeof(int));
	memset(B2, 0, awbh*bw*sizeof(int));
	delete[] buffer, B2;//*/
}

//Naive Matrix Multiplication (obsolete)
void		matrix_multiplication			(int *C, int const *A, int const *B, int ah, int awbh, int bw, int q)
{
	int *B2=new int[awbh*bw];//naive multiplication - cache optimization
	for(int k=0;k<bw;++k)//transpose B
		for(int k2=0;k2<awbh;++k2)
			B2[k*awbh+k2]=B[k2*bw+k];
	memset(C, 0, ah*bw*sizeof(int));
	for(int k=0;k<ah;++k)
	{
		for(int k2=0;k2<bw;++k2)
		{
			auto &Ck=C[k*bw+k2];
		//	Ck=0;
			int const *Ak=A+k*awbh, *Bk=B2+k2*awbh;
			for(int k3=0;k3<awbh;++k3)
				Ck+=Ak[k3]*Bk[k3], Ck%=q;
			//	Ck+=A[k*awbh+k3]*B2[k2*awbh+k3], Ck%=q;
			Ck+=q&-(Ck<0);
		}
	}
	memset(B2, 0, awbh*bw*sizeof(int));
	delete[] B2;
}
void		matrix_multiplication_add		(int *C, int const *A, int const *B, int ah, int awbh, int bw, int q)
{
	int *B2=new int[awbh*bw];//naive multiplication - cache optimization
	for(int k=0;k<bw;++k)//transpose B
		for(int k2=0;k2<awbh;++k2)
			B2[k*awbh+k2]=B[k2*bw+k];
//	memset(C, 0, ah*bw*sizeof(int));
	for(int k=0;k<ah;++k)
	{
		for(int k2=0;k2<bw;++k2)
		{
			auto &Ck=C[k*bw+k2];
			int const *Ak=A+k*awbh, *Bk=B2+k2*awbh;
			for(int k3=0;k3<awbh;++k3)
				Ck+=Ak[k3]*Bk[k3], Ck%=q;
			Ck+=q&-(Ck<0);
		}
	}
	memset(B2, 0, awbh*bw*sizeof(int));
	delete[] B2;
}
void		matrix_multiplication_subtract	(int *C, int const *A, int const *B, int ah, int awbh, int bw, int q)
{
	int *B2=new int[awbh*bw];//naive multiplication - cache optimization
	for(int k=0;k<bw;++k)//transpose B
		for(int k2=0;k2<awbh;++k2)
			B2[k*awbh+k2]=B[k2*bw+k];
//	memset(C, 0, ah*bw*sizeof(int));
	for(int k=0;k<ah;++k)
	{
		for(int k2=0;k2<bw;++k2)
		{
			auto &Ck=C[k*bw+k2];
			int const *Ak=A+k*awbh, *Bk=B2+k2*awbh;
			for(int k3=0;k3<awbh;++k3)
				Ck-=Ak[k3]*Bk[k3], Ck%=q;
			Ck+=q&-(Ck<0);
		}
	}
	memset(B2, 0, awbh*bw*sizeof(int));
	delete[] B2;
}
class		LP11_LWE
{
public:
	static const int l=128, n1=256, n2=256, q=4093;//128+bit security
	static const double s;
public://
	static int R1[n1*l], R2[n2*l], A_bar[n1*n2], P[n1*l];//same for all: A		public key: P	private key: R2
//	static int *R1, *R2, *A_bar, *P;
	static bool encrypt_ready;
	static int e1[n1], e2[n2], e3[l], c1[n2], c2[l];
//	static int *e1, *e2, *e3, *c1, *c2;
public:
	static void Generate()
	{
	//	R1=new int[n1*l], R2=new int[n2*l], A_bar=new int[n1*n2], P=new int[n1*l];
		int g_amp_2=(int)floor(s)/2;
	//	int g_amp_2=(int)floor(s);
		int g_amp=2*g_amp_2;
	//	int fs=(int)floor(s);
		generate_uniform(n1*l*sizeof(int), (unsigned char*)R1);//Gaussian
		for(int k=0, kEnd=n1*l;k<kEnd;++k)
		{
			auto &vk=R1[k];
			vk%=q, vk+=q&-(vk<0), vk%=g_amp, vk-=g_amp_2;
		}
		generate_uniform(n2*l*sizeof(int), (unsigned char*)R2);//Gaussian
		for(int k=0, kEnd=n2*l;k<kEnd;++k)
		{
			auto &vk=R2[k];
			vk%=q, vk+=q&-(vk<0), vk%=g_amp, vk-=g_amp_2;
		}
		generate_uniform(n1*n2*sizeof(int), (unsigned char*)A_bar);//uniform
		for(int k=0, kEnd=n1*n2;k<kEnd;++k)
			A_bar[k]%=q, A_bar[k]+=q&-(A_bar[k]<0);

		memcpy(P, R1, n1*l*sizeof(int));//P = R1 - A R2
		st_subtract=true;
		matrix_multiplication_strassen_add(P, A_bar, R2, n1, n2, l, q);
	//	matrix_multiplication_subtract(P, A_bar, R2, n1, n2, l, q);
		//for(int k=0;k<n1;++k)
		//{
		//	for(int k2=0;k2<l;++k2)
		//	{
		//		auto &Pk=P[k*l+k2];
		//		Pk=(R1[k*l+k2]-Pk)%q, Pk+=q&-(Pk<0);
		//	}
		//}//*/
	/*	for(int k=0;k<n1;++k)//P = R1 - A R2			//naive
		{
			for(int k2=0;k2<l;++k2)
			{
				auto &pk=P[k*l+k2];
				pk=R1[k*l+k2];
				for(int k3=0;k3<n2;++k3)
				{
					pk-=A_bar[k*n2+k3]*R2[k3*l+k2];//R2 needs transpose for speed
					pk%=q, pk+=q&-(pk<0);
				}
			}
		}//*/

	/*	//check P
		int R1_[n1*l]={0};
		for(int k=0;k<n1;++k)
		{
			for(int k2=0;k2<l;++k2)
			{
				auto &vk=R1_[k*l+k2];
				vk=P[k*l+k2];
				for(int k3=0;k3<n2;++k3)
				{
					vk+=A_bar[k*n2+k3]*R2[k3*l+k2];
					vk%=q, vk+=q&-(vk<0);
				}
			}
		}//*/
		encrypt_ready=false;
	}
	static void Destroy()
	{
		memset(R1, 0, n1*l*sizeof(int));
		memset(R2, 0, n2*l*sizeof(int));
		memset(A_bar, 0, n1*n2*sizeof(int));
	//	delete[] R1, R2, A_bar;
	}
	static void Encrypt_Prepare()
	{
	//	e1=new int[n1], e2=new int[n2], e3=new int[l], c1=new int[n2], c2=new int[l];
		int g_amp_2=(int)floor(s)/2;
	//	int g_amp_2=(int)floor(s);
		int g_amp=2*g_amp_2;
		generate_uniform(n1*sizeof(int), (unsigned char*)e1);//Gaussian
		for(int k=0;k<n1;++k)
		{
			auto &vk=e1[k];
			vk%=q, vk+=q&-(vk<0), vk%=g_amp, vk-=g_amp_2;
		}
		generate_uniform(n2*sizeof(int), (unsigned char*)e2);//Gaussian
		for(int k=0;k<n2;++k)
		{
			auto &vk=e2[k];
			vk%=q, vk+=q&-(vk<0), vk%=g_amp, vk-=g_amp_2;
		}
		generate_uniform(l*sizeof(int), (unsigned char*)e3);//Gaussian
		for(int k=0;k<l;++k)
		{
			auto &vk=e3[k];
			vk%=q, vk+=q&-(vk<0), vk%=g_amp, vk-=g_amp_2;
		}

		memcpy(c1, e2, n2*sizeof(int));//c1 = e1 A + e2
	//	st_subtract=false;
	//	matrix_multiplication_strassen_add(c1, e1, A_bar, 1, n1, n2, q);//matrix by vector
		matrix_multiplication_add(c1, e1, A_bar, 1, n1, n2, q);
	//	for(int k=0;k<n2;++k)
	//		c1[k]+=e2[k], c1[k]%=q, c1[k]+=q&-(c1[k]<0);

		memcpy(c2, e3, l*sizeof(int));//c2 = e1 P + e3 + encode(m)
	//	st_subtract=false;
	//	matrix_multiplication_strassen_add(c2, e1, P, 1, n1, l, q);
		matrix_multiplication_add(c2, e1, P, 1, n1, l, q);
	//	for(int k=0;k<l;++k)
	//		c2[k]+=e3[k], c2[k]%=q, c2[k]+=q&-(c2[k]<0);//*/

	/*	for(int k=0;k<n2;++k)//c1 = e1 A + e2					//naive
		{
			auto &ck=c1[k];
			ck=e2[k];
			for(int k2=0;k2<n1;++k2)
			{
				ck+=e1[k2]*A_bar[k2*n2+k];//needs transpose for speed
				ck%=q, ck+=q&-(ck<0);
			}
		}
		for(int k=0;k<l;++k)//c2 = e1 P + e3 + encode(m)		//naive
		{
			auto &ck=c2[k];
			ck=e3[k];
			for(int k2=0;k2<n1;++k2)
			{
				ck+=e1[k2]*P[k2*l+k];//needs transpose for speed
				ck%=q, ck+=q&-(ck<0);
			}
		}//*/
		encrypt_ready=true;
	}
//	static void Encrypt(unsigned char *m, int *&_c1, int *&_c2)
	static void Encrypt(unsigned char *m, int *_c1, int *_c2)//l=128bit message		(n2=256)+l word = 6144bit cipher
//	static void Encrypt(unsigned char m[l/8], int *_c1, int *_c2)
	{
	//	int *encode_m=new int[l];
		int encode_m[l]={0};
		int m_amp=q/2;
		for(int k=0;k<l;++k)
			encode_m[k]=m_amp&-((m[k/8]>>(k%8))&1);
		if(!encrypt_ready)
			Encrypt_Prepare();
		encrypt_ready=false;

	/*	printf("\nencode(m)\n");
		for(int k=0;k<8;++k)
			printf("\t%4d", k);
		printf("\n\n");
		for(int k=0;k<16;++k)
		{
			printf("%3d:\t%4d", k*8, encode_m[k*8]);
			for(int k2=1;k2<8;++k2)
				printf("\t%4d", encode_m[k*8+k2]);
			printf("\n");
		}//*/

		for(int k=0;k<l;++k)
			c2[k]+=encode_m[k];
	//	delete[] encode_m;
	//	_c1=c1, c1=0;
	//	_c2=c2, c2=0;
		memcpy(_c1, c1, n2*sizeof(int));
		memcpy(_c2, c2, l*sizeof(int));
		memset(c1, 0, n2*sizeof(int));
		memset(c2, 0, l*sizeof(int));
	}
	static void Decrypt(int *_c1, int *_c2, unsigned char *m)
//	static void Decrypt(int *_c1, int *_c2, unsigned char m[l/8])
	{
	//	c1=_c1, c2=_c2;
		memcpy(c1, _c1, n2*sizeof(int));
		memcpy(c2, _c2, l*sizeof(int));

	//	int *plain=new int[l];
		int plain[128]={0};
		int m_amp=q/2;

		memcpy(plain, c2, l*sizeof(int));		//plain = c2 + c1 R2
	//	st_subtract=false;
	//	matrix_multiplication_strassen_add(plain, c1, R2, 1, n2, l, q);
		matrix_multiplication_add(plain, c1, R2, 1, n2, l, q);
		for(int k=0;k<l;++k)
			plain[k]-=q&-(plain[k]>m_amp);//make small//*/
	/*	for(int k=0;k<l;++k)//plain = c2 + c1 R2
		{
			auto &pk=plain[k];
			pk=c2[k];
			for(int k2=0;k2<n2;++k2)
			{
				pk+=c1[k2]*R2[k2*l+k];//needs transpose for speed
				pk%=q, pk+=q&-(pk<0);
			}
			pk-=q&-(pk>m_amp);//make small
		}//*/

	/*	printf("\nplain = c1 R2 + c2\n");
		for(int k=0;k<8;++k)
			printf("\t%4d", k);
		printf("\n\n");
		for(int k=0;k<16;++k)
		{
			printf("%3d:\t%4d", k*8, plain[k*8]);
			for(int k2=1;k2<8;++k2)
				printf("\t%4d", plain[k*8+k2]);
			printf("\n");
		}//*/

		int dec_amp=m_amp/2;
		if(m==nullptr)
			m=new unsigned char[16];
		memset(m, 0, (l/8)*sizeof(unsigned char));
		for(int k=0;k<l;++k)
			m[k/8]|=(abs(plain[k])>dec_amp)<<(k%8);
	//	delete[] plain;
	//	delete[] c1, c2;
	}
};
double const LP11_LWE::s=8.35;
int			LP11_LWE::R1[n1*l]={0}, LP11_LWE::R2[n2*l]={0}, LP11_LWE::A_bar[n1*n2]={0}, LP11_LWE::P[n1*l]={0};
//int		*LP11_LWE::R1=0, *LP11_LWE::R2=0, *LP11_LWE::A_bar=0, *LP11_LWE::P=0;
bool		LP11_LWE::encrypt_ready=false;
int			LP11_LWE::e1[n1]={0}, LP11_LWE::e2[n2]={0}, LP11_LWE::e3[l]={0}, LP11_LWE::c1[n2]={0}, LP11_LWE::c2[l]={0};
//int		*LP11_LWE::e1=0, *LP11_LWE::e2=0, *LP11_LWE::e3=0, *LP11_LWE::c1=0, *LP11_LWE::c2=0;

const double v2_s=35.77;
const int	v2_N=320, v2_q=590921, v2_amp=int(v2_s);//126bit security		Portable implementation using Javascript - Kiyomoto, Miyake, Takagi
bool		negative_wrapped=true;
struct		Zq_xn_1//Zq[x]/(x^n-1)		q around 590921=0x90449
{
	int *v;//ascending powers
	Zq_xn_1()
	{
		v=new int[v2_N];
		memset(v, 0, v2_N*sizeof(int));
	}
	Zq_xn_1(Zq_xn_1 const &x2)//copy constructor
	{
		v=new int[v2_N];
		memcpy(v, x2.v, v2_N*sizeof(int));
	}
	Zq_xn_1(Zq_xn_1 &&x2){v=x2.v, x2.v=0;}//move constructor
	~Zq_xn_1()
	{
		memset(v, 0, v2_N*sizeof(int));
		delete[] v;
	}
	void generate_rand()
	{
		generate_uniform(v2_N*sizeof(int), (unsigned char*)v);
		for(int k=0;k<v2_N;++k)
		{
			auto &x=v[k];
			x%=v2_q, x+=v2_q&-(x<0);
		}
	}
	void generate_small(double s)
	{
		generate_uniform(v2_N*sizeof(int), (unsigned char*)v);
		unsigned hamming_masks[]={0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
		for(int k=0;k<v2_N;++k)//NewHope phi_16
		{
			auto pm=hamming_masks;
			unsigned a=v[k];
			a=(a&*pm)+(a>>1&*pm), ++pm;
			a=(a&*pm)+(a>>2&*pm), ++pm;
			a=(a&*pm)+(a>>4&*pm), ++pm;
			a=(a&*pm)+(a>>8&*pm);
			//a=(a&pm[0])+(a>>1&pm[0]);
			//a=(a&pm[1])+(a>>2&pm[1]);
			//a=(a&pm[2])+(a>>4&pm[2]);
			//a=(a&pm[3])+(a>>8&pm[3]);
			v[k]=((unsigned short*)&a)[0]-((unsigned short*)&a)[1];
		}
	//	for(int k=0;k<v2_N;++k)
	//		v[k]%=v2_amp;//indistinguishable
		//	v[k]=rand()%3-1;//[-1, 1]
		//	v[k]=(rand()%2)*2-1;//{-1, 1} useless
		//	v[k]=0;
	}
	void make_positive()
	{
		for(int k=0;k<v2_N;++k)
			v[k]%=v2_q, v[k]+=v2_q&-(v[k]<0);
	}
	void make_small2()
	{
		int m=v2_q/2;
		for(int k=0;k<v2_N;++k)
		{
			auto &x=v[k];
			x%=v2_q, x+=(v2_q&-(x<-m))-(v2_q&-(x>m));
		}
	}
	std::string toString_vector()
	{
		std::stringstream sst;
		sst<<'('<<v[v2_N-1];
		for(int k=v2_N-2;k>=0;--k)
			sst<<",\t"<<v[k];
		//	sst<<", "<<v[k];
		sst<<')';
		return sst.str();
	}
	std::string toString()
	{
		std::stringstream sst;
		int printed=0;
		for(int k=v2_N-1;k>1;--k)
		{
			auto &vk=v[k];
			if(vk)
			{
				if(printed)
				{
					sst<<' ';
					if(vk>0)
						sst<<'+';
				}
				if(vk<0)
					sst<<'-';
				if(printed)
					sst<<' ';
				if((vk!=1)&(vk!=-1))
					sst<<(int)abs(vk);
				sst<<"x^"<<k;
			}
			printed|=vk;
		}
		auto pvk=&v[1];
		if(*pvk)
		{
			if(printed)
			{
					sst<<' ';
					if(*pvk>0)
						sst<<'+';
			}
			if(*pvk<0)
				sst<<'-';
			if(printed)
				sst<<' ';
			if((*pvk!=1)&(*pvk!=-1))
				sst<<(int)abs(*pvk);
			sst<<"x";
		}
		printed|=*pvk;
		pvk=&v[0];
		if(*pvk)
		{
			if(printed)
			{
					sst<<' ';
					if(*pvk>0)
						sst<<'+';
			}
			if(*pvk<0)
				sst<<'-';
			if(printed)
				sst<<' ';
			sst<<(int)abs(*pvk);
		}
		printed|=*pvk;
		if(!sst.rdbuf()->in_avail())
			sst<<'0';
		return sst.str();
	}
	void print(){std::cout<<toString_vector();}
	void print_polynomial(){std::cout<<toString();}
	void print_table(char const *name)
	{
		std::cout<<name<<endl;
		for(int k=0;k<10;++k)
			printf("\t%6d", k);
		//	std::cout<<'\t'<<k;
		std::cout<<endl<<endl;
		for(int k=0;k<v2_N;k+=10)
		{
			printf("%3d :", k);
			for(int k2=0;k2<10;++k2)
				printf("\t%6d", v[k+k2]);
			//	std::cout<<'\t'<<v[k+k2];
			std::cout<<endl;
		}
	}
	Zq_xn_1& operator+=(Zq_xn_1 &other)
	{
		for(int k=0;k<v2_N;++k)
			v[k]+=other.v[k];
	//	*this=*this+b;
		return *this;
	}
	Zq_xn_1& operator-=(Zq_xn_1 &other)
	{
		for(int k=0;k<v2_N;++k)
			v[k]-=other.v[k];
	//	*this=*this+b;
		return *this;
	}
	Zq_xn_1& operator*=(Zq_xn_1 const &b)
	{
		Zq_xn_1 a(std::move(*this));
		v=new int[v2_N];
		for(int k=0;k<v2_N;++k)
		{
			auto &vk=v[v2_N-1-k];
			vk=0;
			int range1=v2_N-1-k;
			for(int k2=range1;k2>=0;--k2)
				vk+=a.v[k2]*b.v[range1-k2]%v2_q;
			for(int k2=v2_N-1;k2>range1;--k2)
				vk-=(a.v[k2]*b.v[v2_N-k2+range1]^-(int)negative_wrapped)+negative_wrapped, vk%=v2_q, vk+=v2_q&-(vk<0);
			//	vk-=a.v[k2]*b.v[v2_N-k2+range1], vk%=v2_q, vk+=v2_q&-(vk<0);//-=: x^4+1
			//	vk+=a.v[k2]*b.v[v2_N-k2+range1], vk%=v2_q;					//+=: x^4-1
		}
		return *this;
	}
	Zq_xn_1& operator*=(Zq_xn_1 b)
	{
		Zq_xn_1 a(std::move(*this));
		v=new int[v2_N];
		for(int k=0;k<v2_N;++k)
		{
			auto &vk=v[v2_N-1-k];
			vk=0;
			int range1=v2_N-1-k;
			for(int k2=range1;k2>=0;--k2)
				vk+=a.v[k2]*b.v[range1-k2]%v2_q;
			for(int k2=v2_N-1;k2>range1;--k2)
				vk-=(a.v[k2]*b.v[v2_N-k2+range1]^-(int)negative_wrapped)+negative_wrapped, vk%=v2_q, vk+=v2_q&-(vk<0);
			//	vk-=a.v[k2]*b.v[v2_N-k2+range1], vk%=v2_q, vk+=v2_q&-(vk<0);//-=: x^4+1
			//	vk+=a.v[k2]*b.v[v2_N-k2+range1], vk%=v2_q;					//+=: x^4-1
		}
		return *this;
	}
};
Zq_xn_1		operator+(Zq_xn_1 &x, Zq_xn_1 &y)
{
	Zq_xn_1 sum;
	for(int k=0;k<v2_N;++k)
	{
		auto &sk=sum.v[k];
		sk=(x.v[k]+y.v[k])%v2_q, sk+=v2_q&-(sk<0);
	}
	return sum;
}
Zq_xn_1		operator-(Zq_xn_1 &x, Zq_xn_1 &y)
{
	Zq_xn_1 sum;
	for(int k=0;k<v2_N;++k)
	{
		auto &sk=sum.v[k];
		sk=(x.v[k]-y.v[k])%v2_q, sk+=v2_q&-(sk<0);
	}
	return sum;
}
Zq_xn_1		operator*(Zq_xn_1 &x, Zq_xn_1 &y)
{
	Zq_xn_1 pr;
//	multiply_polynomials(x.v, y.v, pr.v, v2_N, v2_q, 34, 29073, negative_wrapped);
	//for(int k=0, kEnd=v2_N*2-1;k<kEnd;++k)//convolution
	//{
	//	auto &vk=pr.v[k%v2_N];
	//	int moving_start=k-(v2_N-1);
	//	for(int k2=(moving_start+abs(moving_start))/2, k2End=(k+1+v2_N-abs(k+1-v2_N))/2;k2<k2End;++k2)
	//		vk+=x.v[k2]*y.v[k2End-1-k2]%v2_q, vk+=q&-(vk<0);
	//}

	for(int k=0;k<v2_N;++k)
	{
		auto &vk=pr.v[v2_N-1-k];
		vk=0;
		int range1=v2_N-1-k;
		for(int k2=range1;k2>=0;--k2)
			vk+=x.v[k2]*y.v[range1-k2]%v2_q;
		for(int k2=v2_N-1;k2>range1;--k2)
			vk-=(x.v[k2]*y.v[v2_N-k2+range1]^-(int)negative_wrapped)+negative_wrapped, vk%=v2_q, vk+=v2_q&-(vk<0);//-=: x^4+1
		//	vk-=x.v[k2]*y.v[v2_N-k2+range1], vk%=v2_q, vk+=v2_q&-(vk<0);//-=: x^4+1
		//	vk+=x.v[k2]*y.v[v2_N-k2+range1], vk%=v2_q;					//+=: x^4-1
	}//*/
//	pr.a3=(x.a3*y.a0+x.a2*y.a1+x.a1*y.a2+x.a0*y.a3)%q, pr.a3+=q&-(pr.a3<0);
//	pr.a2=(x.a2*y.a0+x.a1*y.a1+x.a0*y.a2-x.a3*y.a3)%q, pr.a2+=q&-(pr.a2<0);
//	pr.a1=(x.a1*y.a0+x.a0*y.a1-x.a3*y.a2-x.a2*y.a3)%q, pr.a1+=q&-(pr.a1<0);
//	pr.a0=(x.a0*y.a0-x.a3*y.a1-x.a2*y.a2-x.a1*y.a3)%q, pr.a0+=q&-(pr.a0<0);
	return pr;
}
Zq_xn_1		operator*(int x, Zq_xn_1 &y)
{
	Zq_xn_1 pr;
	for(int k=0;k<v2_N;++k)
	{
		auto &vk=pr.v[k];
		vk=x*y.v[k]%v2_q, vk+=v2_q&-(vk<0);
	}
	return pr;
}
Zq_xn_1		operator*(Zq_xn_1 &x, int y)
{
	Zq_xn_1 pr;
	for(int k=0;k<v2_N;++k)
	{
		auto &vk=pr.v[k];
		vk=x.v[k]*y%v2_q, vk+=v2_q&-(vk<0);
	}
	return pr;
}
Zq_xn_1&	operator*=(Zq_xn_1 &a, int b)
{
	for(int k=0;k<v2_N;++k)
		a.v[k]*=b, a.v[k]%=v2_q;
	return a;
}

struct		NewHope_private_key
{
	short *s_ntt;
};
struct		NewHope_public_key
{
	unsigned char *seed;
	short *b_ntt;
};
struct		NewHope_ciphertext
{
	short *u_ntt,
		*v_dash;
};
void		newhope_generate(NewHope_private_key &k_pr, NewHope_public_key &k_pu, NTT_params const &p)
{
	const int align=sizeof(SIMD_type), size=32;
	const short n=p.n;
#ifdef PROFILER
	std::cout<<"KeyGen()\n";
	long long t1=__rdtsc();
#endif
	k_pu.seed=new unsigned char[size];
	generate_uniform(size, k_pu.seed);
#ifdef PROFILER
	std::cout<<"generate_uniform:  "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	short *a_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	FIPS202_SHAKE128(k_pu.seed, size, (unsigned char*)a_ntt, n*sizeof(short));
//	for(int k=0;k<n;++k)a_ntt[k]=2;//
//	std::cout<<"a_ntt:", print_element(a_ntt, n, q);//
#ifdef PROFILER
	std::cout<<"shake a_ntt:       "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	int se_size=n*2;
	short *se_buffer=(short*)_aligned_malloc(se_size*sizeof(short), align),
		*s=se_buffer, *e=se_buffer+n;
	newhope_generate_binomial_16(se_buffer, se_size);
//	memset(se_buffer, 0, se_size*sizeof(short));//
//	short *s=(short*)_aligned_malloc(n*sizeof(short), align);
//	newhope_generate_binomial_16(s, n);//
//	for(int k=0;k<n;++k)s[k]=1;//
#ifdef PROFILER
	std::cout<<"generate s, e:     "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	k_pr.s_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_NTT(s, k_pr.s_ntt, p, false);
//	std::cout<<"s_ntt:", print_element(s_ntt, n, q);//

//	short *e=(short*)_aligned_malloc(n*sizeof(short), align);
//	newhope_generate_binomial_16(e, n);//
//	for(int k=0;k<n;++k)e[k]=1;//
#ifdef PROFILER
	std::cout<<"ntt(s):            "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	k_pu.b_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_NTT(e, k_pu.b_ntt, p, false);
//	std::cout<<"e_ntt:", print_element(b_ntt, n, q);//
#ifdef PROFILER
	std::cout<<"b = ntt(e):        "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	multiply_ntt(k_pu.b_ntt, a_ntt, k_pr.s_ntt, p);//b_ntt = a_ntt*s_ntt + e_ntt
//	std::cout<<"b_ntt = a_ntt*s_ntt + e_ntt:", print_element(k_pu.b_ntt, n, q);//
	_aligned_free(a_ntt), _aligned_free(se_buffer);
//	_aligned_free(a_ntt), _aligned_free(s), _aligned_free(e);
#ifdef PROFILER
	std::cout<<"bn = an sn + en:   "<<__rdtsc()-t1<<endl;
#endif
}
void		newhope_encrypt(const char *message, NewHope_ciphertext &ct, NewHope_public_key const &k_pu, unsigned char *e_seed, NTT_params const &p)
{
	const int align=sizeof(SIMD_type), size=32;
	const short n=p.n, q=p.q;
#ifdef PROFILER
	std::cout<<"Encrypt()\n";
	long long t1=__rdtsc();
#endif
	short *a_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	FIPS202_SHAKE128(k_pu.seed, size, (unsigned char*)a_ntt, n*sizeof(short));
//	for(int k=0;k<n;++k)a_ntt[k]=2;//
#ifdef PROFILER
	std::cout<<"shake a_ntt:       "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	short *buffer=(short*)_aligned_malloc(6*n*sizeof(short), align),
		*s2=buffer, *e1=buffer+n, *e2=buffer+2*n;
	FIPS202_SHAKE128(e_seed, size, (unsigned char*)buffer, 6*n*sizeof(short));
#ifdef PROFILER
	std::cout<<"shake s2, e1, e2:  "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	newhope_convert_binomial_16((int*)buffer, buffer, 3*n);
	//newhope_generate_binomial_16(s2, n);//
	//newhope_generate_binomial_16(e1, n);//
	//newhope_generate_binomial_16(e2, n);//
	//for(int k=0;k<n;++k)s2[k]=1;//
	//for(int k=0;k<n;++k)e1[k]=1;//
	//for(int k=0;k<n;++k)e2[k]=1;//
#ifdef PROFILER
	std::cout<<"binom s2, e1, e2:  "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	short *s2_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_NTT(s2, s2_ntt, p, false);
#ifdef PROFILER
	std::cout<<"ntt(s2):           "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	ct.u_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_NTT(e1, ct.u_ntt, p, false);//u_ntt = a_ntt*s2_ntt + e1_ntt
	//std::cout<<"e1_ntt:", print_element(u_ntt, n, q);//
	//std::cout<<"s2_ntt:", print_element(s2_ntt, n, q);//
	//std::cout<<"a_ntt:", print_element(a_ntt, n, q);//
#ifdef PROFILER
	std::cout<<"u = ntt(e1):       "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	multiply_ntt(ct.u_ntt, a_ntt, s2_ntt, p);
//	std::cout<<"u_ntt = a_ntt*s2_ntt + e1_ntt:", print_element(u_ntt, n, q);//
#ifdef PROFILER
	std::cout<<"un = an s2n + e1n: "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
			
	short *m=(short*)_aligned_malloc(n*sizeof(short), align);
	if(n==1024)
		for(int kb=0, q_2=(q>>1)+1;kb<256;++kb)
			m[kb]=m[kb+256]=m[kb+512]=m[kb+768] = q_2&-(message[kb>>3]>>(kb&7)&1);
	else if(n==512)
		for(int kb=0, q_2=(q>>1)+1;kb<256;++kb)
			m[kb]=m[kb+256]						= q_2&-(message[kb>>3]>>(kb&7)&1);
	else
		for(int kb=0, q_2=(q>>1)+1;kb<256;++kb)
			m[kb]								= q_2&-(message[kb>>3]>>(kb&7)&1);
	//for(int kc=0, q_2=(q>>1)+1;kc<size;++kc)
	//{
	//	for(int kb=0;kb<8;++kb)
	//	{
	//		int kx=(kc<<3)|kb;
	//		m[kx]=m[kx+256]=m[kx+512]=m[kx+768] = q_2&-(message[kc]>>kb&1);
	//	}
	//}
	//for(int kv=0, q_2=(q>>1)+1;kv<n;++kv)	//CRASH m: 256, n=1024
	//	m[kv]=q_2*(message[kv>>3]>>(kv&7)&1);
//	std::cout<<"m:", print_element(m, n, q);//
#ifdef PROFILER
	std::cout<<"encode(m):         "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	short *temp=(short*)_aligned_malloc(n*sizeof(short), align);
	memset(temp, 0, n*sizeof(short));
	multiply_ntt(temp, k_pu.b_ntt, s2_ntt, p);//v' = INTT(b_ntt*s2_ntt) + e2 + m
#ifdef PROFILER
	std::cout<<"v'n = bn s2n:      "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	ct.v_dash=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_inverse_NTT(temp, ct.v_dash, p);
#ifdef PROFILER
	std::cout<<"intt(v'n):         "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	for(int kx=0;kx<n;++kx)
	{
		auto &vk=ct.v_dash[kx];
		vk+=e2[kx], vk-=q&-(vk>q);
		vk+=m[kx], vk-=q&-(vk>q);
	}
//	std::cout<<"v' = INTT(b_ntt*s2_ntt) + e2 + m:", print_element(v_dash, n, q);//
	_aligned_free(a_ntt), _aligned_free(buffer), _aligned_free(s2_ntt), _aligned_free(m), _aligned_free(temp);
#ifdef PROFILER
	std::cout<<"v' += e2 - enc(m): "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
}
void		newhope_decrypt(NewHope_ciphertext const &ct, char *message2, NewHope_private_key const &k_pr, NTT_params const &p)
{
	const int align=sizeof(SIMD_type), size=32;
	const short n=p.n, q=p.q;
#ifdef PROFILER
	std::cout<<"Decrypt()\n";
	long long t1=__rdtsc();
#endif
	short *temp=(short*)_aligned_malloc(n*sizeof(short), align);
	memset(temp, 0, n*sizeof(short));
	multiply_ntt(temp, ct.u_ntt, k_pr.s_ntt, p);//m2 = v' - INTT(u_ntt*s_ntt)
#ifdef PROFILER
	std::cout<<"tn = un sn:         "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	short *m2=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_inverse_NTT(temp, m2, p);
#ifdef PROFILER
	std::cout<<"m2 = intt(tn):      "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	
	memset(message2, 0, size);
	for(int kx=0;kx<n;++kx)
	{
		auto &vk=m2[kx];
		vk=ct.v_dash[kx]-vk, vk+=q&-(vk<0);
	}
//	std::cout<<"m2 = v' - INTT(u_ntt*s_ntt):", print_element(m2, n, q);//
#ifdef PROFILER
	std::cout<<"m2 = v' - m2:       "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif

	//if(n==1024)
	//	for(int kb=0, q_2=q>>1;kb<256;++kb)
	//	{
	//		int t=abs(m2[kb]-q_2)+abs(m2[kb+256]-q_2)+abs(m2[kb+512]-q_2)+abs(m2[kb+768]-q_2);
	//		t=(t-q)>>15;
	//		message2[kb>>3]|=t<<(kb&7);
	//	}
	//else if(n==512)
	//	for(int kb=0, q_2=q>>1;kb<256;++kb)
	//	{
	//		int t=abs(m2[kb]-q_2)+abs(m2[kb+256]-q_2);
	//		t=(t-q_2)>>15;
	//		message2[kb>>3]|=t<<(kb&7);
	//	}
	//else
		//for(int kb=0, q_2=q>>1;kb<256;++kb)
		//{
		//	int t=abs(m2[kb]-q_2);
		//	message2[kb>>3]|=t<<(kb&7);
		//}
		for(int kb=0, q_4=q/4, q3_4=q*3/4;kb<256;++kb)
		{
			int t=(m2[kb]>q_4)&(m2[kb]<q3_4);
			message2[kb>>3]|=t<<(kb&7);
		}
	_aligned_free(temp), _aligned_free(m2);
#ifdef PROFILER
	std::cout<<"decode(m2):         "<<__rdtsc()-t1<<endl;
#endif
}
void		newhope_cpa_encapsulate(NewHope_public_key const &pu_k, NewHope_ciphertext &ct, unsigned char *K, NTT_params const &p)
{
	const int size=32;

	unsigned char *e_seed=new unsigned char[size];
	generate_uniform(size, e_seed);
	int buf_size=size<<1;
	unsigned char *buffer=new unsigned char[buf_size],
		*key=buffer, *seed_dash=buffer+size;
	FIPS202_SHAKE256(e_seed, size, buffer, buf_size);
	delete[] e_seed;
//	std::cout<<"seed':\t", print_buffer(seed_dash, size);//
//	std::cout<<"key:\t", print_buffer(key, size);//

	newhope_encrypt((char*)key, ct, pu_k, seed_dash, p);

//	K=new unsigned char[size];
//	FIPS202_SHAKE256(key, size, K, size);
	memcpy(K, key, size);//
	delete[] buffer;
}
void		newhope_cpa_decapsulate(NewHope_private_key const &pr_k, NewHope_public_key const &pu_k, NewHope_ciphertext const &ct, unsigned char *K2, NTT_params const &p)
{
	const int size=32;

	unsigned char *key2=new unsigned char[size];
	newhope_decrypt(ct, (char*)key2, pr_k, p);
//	std::cout<<"key2:\t", print_buffer(key2, size);//

//	FIPS202_SHAKE256(key2, size, K2, size);
	memcpy(K2, key2, size);//
	delete[] key2;
}
struct NewHope_KE_private_key
{
	NewHope_private_key k_pr;
	NewHope_public_key k_pu;
	unsigned char *hash, *s;
};
struct NewHope_KE_ciphertext
{
	unsigned char *d;
	NewHope_ciphertext ct;
};
void		newhope_cca_generate(NewHope_KE_private_key &ke_pr, NewHope_public_key &pu_k, NTT_params const &p)
{
//	std::cout<<"CCA KeyGen()\n";//
	const int align=sizeof(SIMD_type), size=32;

	newhope_generate(ke_pr.k_pr, pu_k, p);

	ke_pr.k_pu.seed=new unsigned char[size];
	ke_pr.k_pu.b_ntt=(short*)_aligned_malloc(p.n*sizeof(short), align);
	memcpy(ke_pr.k_pu.seed, pu_k.seed, size);
	memcpy(ke_pr.k_pu.b_ntt, pu_k.b_ntt, p.n);

	ke_pr.s=new unsigned char[size];
	generate_uniform(size, ke_pr.s);

	int pu_size=size+p.n*sizeof(short);
	unsigned char *pu_buffer=new unsigned char[pu_size];
	memcpy(pu_buffer, pu_k.seed, size);
	memcpy(pu_buffer+size, pu_k.b_ntt, p.n*sizeof(short));

	ke_pr.hash=new unsigned char[size];
	FIPS202_SHAKE256(pu_buffer, pu_size, ke_pr.hash, size);
//	std::cout<<"h = SHAKE256(pu_k):\t", print_buffer(ke_pr.hash, size);//
}
void		newhope_cca_encapsulate(NewHope_public_key const &pu_k, NewHope_KE_ciphertext &ke_ct, unsigned char *K, NTT_params const &p)
{
//	std::cout<<"\nCCA Encaps()\n";//
	const int size=32;

	unsigned char *seed=new unsigned char[size];
	generate_uniform(size, seed);
	unsigned char *message=new unsigned char[size];
	FIPS202_SHAKE256(seed, size, message, size);
	delete[] seed;
//	std::cout<<"message:\t", print_buffer(message, size);//

	int pu_size=size+p.n*sizeof(short);
	unsigned char *pu_buffer=new unsigned char[size+p.n*sizeof(short)];
	memcpy(pu_buffer, pu_k.seed, size);
	memcpy(pu_buffer+size, pu_k.b_ntt, p.n*sizeof(short));

	int mh_size=size<<1;
	unsigned char *mh_buffer=new unsigned char[size<<1];
	FIPS202_SHAKE256(pu_buffer, pu_size, mh_buffer+size, size);
	delete[] pu_buffer;
	memcpy(mh_buffer, message, size);
//	std::cout<<"h = SHAKE256(pu_k):\t", print_buffer(mh_buffer+size, size);//
//	std::cout<<"mh_buffer:\t", print_buffer(mh_buffer, mh_size);//

	int ksd_size=size*3;
	unsigned char *ksd_buffer=new unsigned char[ksd_size],
		*key=ksd_buffer, *seed_dash=ksd_buffer+size, *d=ksd_buffer+size*2;
	FIPS202_SHAKE256(mh_buffer, mh_size, ksd_buffer, ksd_size);
	delete[] mh_buffer;
//	std::cout<<"key:\t", print_buffer(key, size);//
//	std::cout<<"seed':\t", print_buffer(seed_dash, size);//
//	std::cout<<"d:\t", print_buffer(d, size);//

	ke_ct.d=new unsigned char[size];
	memcpy(ke_ct.d, d, size);

	newhope_encrypt((char*)message, ke_ct.ct, pu_k, seed_dash, p);
	delete[] message;

	int cd_size=p.n*2*sizeof(short)+size;
	unsigned char *cd_buffer=new unsigned char[cd_size];
	memcpy(cd_buffer, ke_ct.ct.u_ntt, p.n*sizeof(short));
	memcpy(cd_buffer+p.n, ke_ct.ct.v_dash, p.n*sizeof(short));
	memcpy(cd_buffer+p.n*2, d, size);

	int kh_size=size<<1;
	unsigned char *kh_buffer=new unsigned char[kh_size];
	FIPS202_SHAKE256(cd_buffer, cd_size, kh_buffer+size, size);
	memcpy(kh_buffer, key, size);
	delete[] cd_buffer, ksd_buffer;
//	std::cout<<"kh_buffer:\t", print_buffer(kh_buffer, kh_size);//

	FIPS202_SHAKE256(kh_buffer, kh_size, K, size);
	delete[] kh_buffer;
}
bool		newhope_cca_decapsulate(NewHope_KE_private_key const &ke_pr, NewHope_public_key const &pu_k, NewHope_KE_ciphertext const &ke_ct, unsigned char *K2, NTT_params const &p)
{
	const int size=32;
//	std::cout<<"\nCCA Decaps()\n";//

	unsigned char *message2=new unsigned char[size];
	newhope_decrypt(ke_ct.ct, (char*)message2, ke_pr.k_pr, p);

	int mh_size=size<<1;
	unsigned char *mh_buffer=new unsigned char[mh_size];
	memcpy(mh_buffer, message2, size);
	memcpy(mh_buffer+size, ke_pr.hash, size);
//	std::cout<<"message2:\t", print_buffer(message2, size);//
//	std::cout<<"h = SHAKE256(k_pu):\t", print_buffer(ke_pr.hash, size);//
//	std::cout<<"mh_buffer:\t", print_buffer(mh_buffer, mh_size);//

	int ksd2_size=size*3;
	unsigned char *ksd2_buffer=new unsigned char[ksd2_size],
		*key2=ksd2_buffer, *seed_dash2=ksd2_buffer+size, *d2=ksd2_buffer+size*2;
	FIPS202_SHAKE256(mh_buffer, mh_size, ksd2_buffer, ksd2_size);
	delete[] mh_buffer;
//	std::cout<<"key2:\t", print_buffer(key2, size);//
//	std::cout<<"seed'2:\t", print_buffer(seed_dash2, size);//
//	std::cout<<"d2:\t", print_buffer(d2, size);//

	NewHope_ciphertext ct2;
	newhope_encrypt((char*)message2, ct2, pu_k, seed_dash2, p);
	delete[] message2;

	int success=memcmp(ke_ct.ct.u_ntt, ct2.u_ntt, p.n*sizeof(short))==0;
	success&=memcmp(ke_ct.ct.v_dash, ct2.v_dash, p.n*sizeof(short))==0;
	success&=memcmp(ke_ct.d, d2, size)==0;

	//if(!success)
	//	memcpy(key2, ke_pr.s, size);

	int cd_size=p.n*2*sizeof(short)+size;
	unsigned char *cd_buffer=new unsigned char[cd_size];
	memcpy(cd_buffer, ke_ct.ct.u_ntt, p.n*sizeof(short));
	memcpy(cd_buffer+p.n, ke_ct.ct.v_dash, p.n*sizeof(short));
	memcpy(cd_buffer+p.n*2, ke_ct.d, size);

	int kh_size=size<<1;
	unsigned char *kh_buffer=new unsigned char[kh_size];
	FIPS202_SHAKE256(cd_buffer, cd_size, kh_buffer+size, size);
	delete[] cd_buffer;

	int success_mask=-success;
	void *src=(void*)(success_mask&(int)key2|~success_mask&(int)ke_pr.s);
	memcpy(kh_buffer, src, size);
//	std::cout<<"kh_buffer:\t", print_buffer(kh_buffer, kh_size);//

	FIPS202_SHAKE256(kh_buffer, kh_size, K2, size);
	delete[] ksd2_buffer, kh_buffer;
	std::cout<<endl;//

	return success!=0;
}

struct		Kyber_private_key
{
	short *s_ntt;//768 *13bit
};
struct		Kyber_public_key
{
	unsigned char *rho;//256bit = 32 bytes
	short *t;	//768 *11bit -> 528 shorts = 1056 bytes
};
struct		Kyber_ciphertext
{
	short *u,	//768 *11bit -> 528 shorts = 1056 bytes
		*v;		//256 *3bit -> 48 shorts = 96 bytes
};
struct		Kyber_KE_ciphertext
{
	unsigned char *u,	//1056 bytes
		*v,				//96 bytes
		*d;				//32 bytes
};
void		kyber_generate(Kyber_private_key &pr_k, Kyber_public_key &pu_k, NTT_params &p)
{
	const int align=sizeof(SIMD_type);
	const short n=p.n, q=p.q, size=n>>3;
	const int k=3, A_size=k*k*n, vector_size=k*n;
//	const double _11_q=double(1<<11)/q, _3_q=double(1<<3)/q;

	//Generation
	pu_k.rho=(unsigned char*)_aligned_malloc(size, align);//256bit
	generate_uniform(size, pu_k.rho);

	short *A=(short*)_aligned_malloc(A_size*sizeof(short), align);
	kyber_uniform_rejection_sampling(A, A_size, pu_k.rho, size, q);
//	generate_uniform(A_size*sizeof(short), (unsigned char*)A);//
//	for(int kv=0;kv<A_size;++kv)A[kv]=1;//
//	memset(A, 0, A_size*sizeof(short));//
//	std::cout<<"A:", print_element(A, A_size, q);//
//	std::cout<<"A histogram", print_histogram(A, A_size, q);//
	
	unsigned char *sigma=(unsigned char*)_aligned_malloc(size, align);
	generate_uniform(size, sigma);
	short *se_buffer=(short*)_aligned_malloc(2*vector_size*sizeof(short), align),
		*s=se_buffer, *e=se_buffer+vector_size;
	FIPS202_SHAKE128(sigma, size, (unsigned char*)se_buffer, 2*vector_size*sizeof(short));
	kyber_convert_binomial_4(se_buffer, 2*vector_size);
//	for(int kv=0;kv<2*vector_size;++kv)se_buffer[kv]=1;//
//	memset(se_buffer, 0, 2*vector_size*sizeof(short));//
	//short *s=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	//short *e=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	//kyber_generate_binomial_4(s, vector_size);
	//kyber_generate_binomial_4(e, vector_size);
//	memset(s, 0, vector_size*sizeof(short));//
//	memset(e, 0, vector_size*sizeof(short));//

	pr_k.s_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	short *e_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_NTT(s+n*kx, pr_k.s_ntt+n*kx, p, false);//no BRP
	for(int kx=0;kx<k;++kx)
		apply_NTT(e+n*kx, e_ntt+n*kx, p, false);//no BRP

	memset(se_buffer, 0, 2*vector_size*sizeof(short));//security measure
	//memset(s, 0, vector_size*sizeof(short));
	//memset(e, 0, vector_size*sizeof(short));
	_aligned_free(sigma);
	_aligned_free(se_buffer);
//	_aligned_free(s), _aligned_free(e);

	//t = A s + e
	short *t_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	memcpy(t_ntt, e_ntt, vector_size*sizeof(short));
//	memset(t_ntt, 0, vector_size*sizeof(short));
//	std::cout<<"A:"; for(int kx=0;kx<k*k;++kx)print_element(A+n*kx, n, q);//
//	std::cout<<"s_ntt:"; for(int kx=0;kx<k;++kx)print_element(pr_k.s_ntt+n*kx, n, q);//
	for(int ky=0;ky<k;++ky)
		for(int kx=0;kx<k;++kx)
			multiply_ntt_add(t_ntt+n*ky, A+n*(k*ky+kx), pr_k.s_ntt+n*kx, p);
			//for(int kv=0;kv<n;++kv)
			//	t_ntt[n*ky+kv]=kyber_montgomery_reduction(t_ntt[n*ky+kv]+A[n*(k*ky+kx)+kv]*pr_k.s_ntt[n*kx+kv]);
//	std::cout<<"t_ntt = A.s_ntt + e_ntt:"; for(int kx=0;kx<k;++kx)print_element(t_ntt+n*kx, n, q);//
	//for(int kx=0;kx<k;++kx)
	//	for(int kv=0;kv<n;++kv)
	//		t_ntt[n*kx+kv]=kyber_montgomery_reduction(t_ntt[n*kx+kv]+e_ntt[n*kx+kv]);

	//t=compress(t, 11)
	pu_k.t=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_inverse_NTT(t_ntt+n*kx, pu_k.t+n*kx, p);
	kyber_compress(pu_k.t, vector_size, q, 11);
//	kyber_compress(pu_k.t, vector_size, _11_q);
	short *t_temp=pack_bits((unsigned short*)pu_k.t, vector_size, 11, 16, align);
	_aligned_free(pu_k.t), pu_k.t=t_temp;
	//Public Key: (t, A)	secret key: s_ntt

	_aligned_free(A);
//	_aligned_free(s), _aligned_free(e);
	_aligned_free(e_ntt);
	_aligned_free(t_ntt);
}
void		kyber_destroy(Kyber_private_key &pr_k, Kyber_public_key &pu_k, NTT_params const &p)
{
	memset(pr_k.s_ntt, 0, 3*p.n*sizeof(short));//security measure
	_aligned_free(pr_k.s_ntt);
	_aligned_free(pu_k.rho), _aligned_free(pu_k.t);
}
void		kyber_encrypt(Kyber_public_key const &pu_k, const char *message, const unsigned char *r_seed, Kyber_ciphertext &ct, NTT_params const &p)
{
	const int align=sizeof(SIMD_type);
	const short n=p.n, q=p.q, q_2=p.q/2+1, size=n>>3;
	const int k=3, A_size=k*k*n, vector_size=k*n;
//	const double _11_q=double(1<<11)/q, _3_q=double(1<<3)/q;

	//Encryption
	short *m=(short*)_aligned_malloc(n*sizeof(short), align);
	for(int kv=0;kv<n;++kv)
		m[kv]=q_2&-(message[kv>>3]>>(kv&7)&1);
	//	m[kv]=q_2*(message[kv>>3]>>(kv&7)&1);

	//t=decompress(t, 11)
	short *t=unpack_bits((unsigned short*)pu_k.t, vector_size*11/16, 11, 16, align);
	kyber_decompress(t, vector_size, q, 11);
//	kyber_decompress(t, vector_size, _11_q);
	short *t_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_NTT(t+n*kx, t_ntt+n*kx, p, true);
	_aligned_free(t);
	
	short *A=(short*)_aligned_malloc(A_size*sizeof(short), align);
	kyber_uniform_rejection_sampling(A, A_size, pu_k.rho, size, q);
//	for(int kv=0;kv<A_size;++kv)A[kv]=1;//
//	memset(A, 0, A_size*sizeof(short));//
//	std::cout<<"A:", print_element(A, A_size, q);//

	const int re_size=2*vector_size+n;
	short *re_buffer=(short*)_aligned_malloc(re_size*sizeof(short), align),
		*r=re_buffer, *e1=re_buffer+vector_size, *e2=re_buffer+vector_size*2;
	FIPS202_SHAKE128(r_seed, size, (unsigned char*)re_buffer, re_size*sizeof(short));
	kyber_convert_binomial_4(re_buffer, 2*vector_size+n);
//	memset(re_buffer, 0, re_size*sizeof(short));//

	short *r_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_NTT(r+n*kx, r_ntt+n*kx, p, false);//no BRP

	//u = INTT(AT r) + e1
	short *u_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	memset(u_ntt, 0, vector_size*sizeof(short));
	for(int ky=0;ky<k;++ky)
		for(int kx=0;kx<k;++kx)
			multiply_ntt_add(u_ntt+n*ky, A+n*(k*kx+ky), r_ntt+n*kx, p);
			//for(int kv=0;kv<n;++kv)
			//	u_ntt[n*ky+kv]=kyber_montgomery_reduction(u_ntt[n*ky+kv]+A[n*(k*kx+ky)+kv]*r_ntt[n*kx+kv]);
	ct.u=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_inverse_NTT(u_ntt+n*kx, ct.u+n*kx, p);
	add_polynomials(ct.u, ct.u, e1, vector_size, q);
	//for(int kx=0;kx<k;++kx)
	//	for(int kv=0;kv<n;++kv)
	//		ct.u[n*kx+kv]=kyber_montgomery_reduction(ct.u[n*kx+kv]+e1[n*kx+kv]);

	//v = tT r + e2 + round(q/2)*m
	short *v_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	memset(v_ntt, 0, n*sizeof(short));
	for(int kx=0;kx<k;++kx)
		multiply_ntt_add(v_ntt, t_ntt+n*kx, r_ntt+n*kx, p);
		//for(int kv=0;kv<n;++kv)
		//	v_ntt[kv]=kyber_montgomery_reduction(v_ntt[kv]+t_ntt[n*kx+kv]*r_ntt[n*kx+kv]);
	ct.v=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_inverse_NTT(v_ntt, ct.v, p);
	add_polynomials(ct.v, ct.v, e2, n, q);
	add_polynomials(ct.v, ct.v, m, n, q);
	//for(int kv=0;kv<n;++kv)
	//	ct.v[kv]=kyber_montgomery_reduction(ct.v[kv]+e2[kv]+m[kv]);

	kyber_compress(ct.u, vector_size, q, 11);//11 bit coefficients	//u = compress(u, 11)
//	kyber_compress(ct.u, vector_size, _11_q);
	//encode u	256*3*11 bit -> k*n*11/(8*sizeof(short) = 528 short
/*	int u_size=vector_size*11/(8*sizeof(short));
	short *u2=(short*)_aligned_malloc(u_size*sizeof(short), align);
	short mask=0x07FF;
	for(int k=0, k2=0;k<vector_size;k+=16, k2+=11)
	{
		u2[k2+0]=(ct.u[k+1]&mask)<<11|(ct.u[k+0]&mask);
		u2[k2+1]=(ct.u[k+2]&mask)<<6|(ct.u[k+1]&mask)>>5;
		u2[k2+2]=(ct.u[k+4]&mask)<<12|(ct.u[k+3]&mask)<<1|(ct.u[k+2]&mask)>>10;
		u2[k2+3]=(ct.u[k+5]&mask)<<7|(ct.u[k+4]&mask)>>4;
		u2[k2+4]=(ct.u[k+7]&mask)<<13|(ct.u[k+6]&mask)<<2|(ct.u[k+5]&mask)>>9;
		u2[k2+5]=(ct.u[k+8]&mask)<<8|(ct.u[k+7]&mask)>>3;
		u2[k2+6]=(ct.u[k+10]&mask)<<14|(ct.u[k+9]&mask)<<3|(ct.u[k+8]&mask)>>8;
		u2[k2+7]=(ct.u[k+11]&mask)<<9|(ct.u[k+10]&mask)>>2;
		u2[k2+8]=(ct.u[k+13]&mask)<<15|(ct.u[k+12]&mask)<<4|(ct.u[k+11]&mask)>>7;
		u2[k2+9]=(ct.u[k+14]&mask)<<10|(ct.u[k+13]&mask)>>1;
		u2[k2+10]=(ct.u[k+15]&mask)<<5|(ct.u[k+14]&mask)>>6;
	}
	_aligned_free(ct.u), ct.u=u2;//*/
	ct.u=pack_bits((unsigned short*)ct.u, vector_size, 11, 16, align);
	
	kyber_compress(ct.v, n, q, 3);//3 bit coefficients			//v = compress(v, 3)
//	kyber_compress(ct.v, n, _3_q);
	//encode v	256*3 bit -> n*3/(8*sizeof(short)) = 48 short
/*	int v_size=n*3/(8*sizeof(short));
	unsigned char *cv=(unsigned char*)ct.v;
	unsigned char mask=7;
	for(int k=0, k2=0;k<n;k+=8, k2+=3)//each 8 coefficients -> 21 bit = 3 bytes
	{
		cv[k2+0]=					 (ct.v[k+2]&mask)<<6|(ct.v[k+1]&mask)<<3|(ct.v[k+0]&mask);
		cv[k2+1]=(ct.v[k+5]&mask)<<7|(ct.v[k+4]&mask)<<4|(ct.v[k+3]&mask)<<1|(ct.v[k+2]&mask)>>2;
		cv[k2+2]=					 (ct.v[k+7]&mask)<<5|(ct.v[k+6]&mask)<<2|(ct.v[k+5]&mask)>>1;
	}
	ct.v=(short*)_aligned_realloc(ct.v, v_size*sizeof(short), align);//*/
	ct.v=pack_bits((unsigned short*)ct.v, n, 3, 16, align);
	
	memset(re_buffer, 0, re_size*sizeof(short));
	_aligned_free(re_buffer);
	_aligned_free(m);//, _aligned_free(m_ntt);
	_aligned_free(A);
	_aligned_free(r_ntt);//, _aligned_free(e1_ntt), _aligned_free(e2_ntt);
	_aligned_free(u_ntt), _aligned_free(v_ntt);
}
void		kyber_encrypt(const char *message, Kyber_ciphertext &ct, Kyber_public_key const &pu_k, NTT_params const &p)
{
	const int align=sizeof(SIMD_type);
	const short n=p.n, q=p.q, q_2=p.q/2+1, size=n>>3;

	unsigned char *r_seed=(unsigned char*)_aligned_malloc(size, align);
	generate_uniform(size, r_seed);
//	std::cout<<"\npu_k.t:\n", print_buffer(pu_k.t, 1056);//
	kyber_encrypt(pu_k, message, r_seed, ct, p);
//	std::cout<<"\npu_k.t:\n", print_buffer(pu_k.t, 1056);//
	_aligned_free(r_seed);

/*	const int align=sizeof(SIMD_type);
	const short n=p.n, q=p.q, q_2=p.q/2+1;
	const int k=3, A_size=k*k*n, vector_size=k*n;
	const double _11_q=double(1<<11)/q, _3_q=double(1<<3)/q;

	//Encryption
//	const char *message="12345678901234567890123456789012";//256bit
//	std::cout<<"Message:\t"<<message<<endl;
	short *m=(short*)_aligned_malloc(n*sizeof(short), align);
	for(int kv=0;kv<n;++kv)
		m[kv]=q_2*(message[kv>>3]>>(kv&7)&1);
	short *m_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_NTT(m, m_ntt, p, true);

	//t=decompress(t, 11)
	kyber_decompress(pu_k.t, vector_size, _11_q);
	short *t_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_NTT(pu_k.t+n*kx, t_ntt+n*kx, p, true);
	
	short *A=(short*)_aligned_malloc(A_size*sizeof(short), align);
	kyber_uniform_rejection_sampling(A, A_size, pu_k.rho, n>>3, q);
		//unsigned char *r_src=(unsigned char*)_aligned_malloc(n>>3, align);//256bit
		//generate_uniform(n>>3, r_src);
	
	unsigned char *r_seed=(unsigned char*)_aligned_malloc(n>>3, align);
	generate_uniform(n>>3, r_seed);
	const int re_size=2*vector_size+n;
	short *re_buffer=(short*)_aligned_malloc(re_size*sizeof(short), align);
	FIPS202_SHAKE128(r_seed, n>>3, (unsigned char*)re_buffer, re_size*sizeof(short));
	kyber_convert_binomial_4(re_buffer, 2*vector_size+n);
	short *r=re_buffer, *e1=re_buffer+vector_size, *e2=re_buffer+vector_size*2;
	//short *r=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	//short *e1=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	//short *e2=(short*)_aligned_malloc(n*sizeof(short), align);
	//kyber_generate_binomial_4(r, vector_size);
	//kyber_generate_binomial_4(e1, vector_size);
	//kyber_generate_binomial_4(e2, n);
//	memset(r, 0, vector_size*sizeof(short));//
//	memset(e1, 0, vector_size*sizeof(short));//
//	memset(e2, 0, n*sizeof(short));//
	short *r_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	short *e1_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	short *e2_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_NTT(r+n*kx, r_ntt+n*kx, p, false);//no BRP
	for(int kx=0;kx<k;++kx)
		apply_NTT(e1+n*kx, e1_ntt+n*kx, p, false);//no BRP
	apply_NTT(e2, e2_ntt, p, false);			//no BRP
	memset(re_buffer, 0, re_size*sizeof(short));//security measure
	//memset(r, 0, vector_size*sizeof(short));
	//memset(e1, 0, vector_size*sizeof(short));
	//memset(e2, 0, n*sizeof(short));
	_aligned_free(r_seed);
	_aligned_free(re_buffer);
//	_aligned_free(r), _aligned_free(e1), _aligned_free(e2);

	//u = AT r + e1
	short *u_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	memset(u_ntt, 0, vector_size*sizeof(short));
	for(int ky=0;ky<k;++ky)
		for(int kx=0;kx<k;++kx)
			for(int kv=0;kv<n;++kv)
				u_ntt[n*ky+kv]=kyber_montgomery_reduction(u_ntt[n*ky+kv]+A[n*(k*kx+ky)+kv]*r_ntt[n*kx+kv]);
	for(int kx=0;kx<k;++kx)
		for(int kv=0;kv<n;++kv)
			u_ntt[n*kx+kv]=kyber_montgomery_reduction(u_ntt[n*kx+kv]+e1_ntt[n*kx+kv]);

	//v = tT r + e2 + round(q/2)*m
	short *v_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	memset(v_ntt, 0, n*sizeof(short));
	for(int kx=0;kx<k;++kx)
		for(int kv=0;kv<n;++kv)
			v_ntt[kv]=kyber_montgomery_reduction(v_ntt[kv]+t_ntt[n*kx+kv]*r_ntt[n*kx+kv]);
	for(int kv=0;kv<n;++kv)
		v_ntt[kv]=kyber_montgomery_reduction(v_ntt[kv]+e2_ntt[kv]+m_ntt[kv]);
			
	ct.u=(short*)_aligned_malloc(vector_size*sizeof(short), align);	//u = compress(u, 11)
	for(int kx=0;kx<k;++kx)
		apply_inverse_NTT(u_ntt+n*kx, ct.u+n*kx, p);
	kyber_compress(ct.u, vector_size, _11_q);//11 bit coefficients
			
	ct.v=(short*)_aligned_malloc(n*sizeof(short), align);			//v = compress(v, 3)
	apply_inverse_NTT(v_ntt, ct.v, p);
	kyber_compress(ct.v, n, _3_q);//3 bit coefficients

	_aligned_free(m), _aligned_free(m_ntt);
	_aligned_free(A);
//	_aligned_free(r), _aligned_free(e1), _aligned_free(e2);
	_aligned_free(r_ntt), _aligned_free(e1_ntt), _aligned_free(e2_ntt);
	_aligned_free(u_ntt), _aligned_free(v_ntt);//*/
}
void		kyber_decrypt(Kyber_ciphertext const &ct, char *message2, Kyber_private_key const &pr_k, NTT_params const &p)
{
	const int align=sizeof(SIMD_type);
	const short n=p.n, q=p.q, q_2=p.q/2+1;
	const int k=3, A_size=k*k*n, vector_size=k*n;
//	const double _11_q=double(1<<11)/q, _3_q=double(1<<3)/q;

	//Decryption
	//decode u
/*	int u_size=vector_size*11/(8*sizeof(short));
	short *u2=ct.u;
	ct.u=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	short mask=0x07FF;
	for(int k=vector_size*11/8-1, k2=vector_size-1;k2>0;k-=11, k2-=16)
	{
		ct.u[k2+0]=u2[k+0]&mask, ct.u[k2+1]=(u2[k+1]<<5|u2[k+0]>>11)&mask, ct.u[k2+2]=(u2[k+2]<<10|u2[k+1]>>6)&mask;
		ct.u[
	}
	_aligned_free(u2);//*/
	short *u=unpack_bits((unsigned short*)ct.u, vector_size*11/16, 11, 16, align);
	kyber_decompress(u, vector_size, q, 11);					//u = decompress(u, 11)
//	kyber_decompress(u, vector_size, _11_q);
	short *u_ntt=(short*)_aligned_malloc(vector_size*sizeof(short), align);
	for(int kx=0;kx<k;++kx)
		apply_NTT(u+n*kx, u_ntt+n*kx, p, true);
	_aligned_free(u);
	
	//decode v
/*	ct.v=(short*)_aligned_realloc(ct.v, n*sizeof(short), align);
	unsigned char *cv=(unsigned char*)ct.v;
	unsigned char mask=7;
	for(int k=n*3/8-1, k2=n-1;k2>0;k-=3, k2-=8)
	{
		ct.v[k2-0]=cv[k-0]>>5&mask, ct.v[k2-1]=cv[k-0]>>2&mask, ct.v[k2-2]=(cv[k-0]<<1|cv[k-1]>>7)&mask;
		ct.v[k2-3]=cv[k-1]>>4&mask, ct.v[k2-4]=cv[k-1]>>1&mask, ct.v[k2-5]=(cv[k-1]<<2|cv[k-2]>>6)&mask;
		ct.v[k2-6]=cv[k-2]>>3&mask, ct.v[k2-7]=cv[k-2]&mask;
	}//*/
	short *v=unpack_bits((unsigned short*)ct.v, n*3/16, 3, 16, align);

	kyber_decompress(v, n, q, 3);							//v = decompress(v, 3)
//	kyber_decompress(v, n, _3_q);
	short *v_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_NTT(v, v_ntt, p, true);
	_aligned_free(v);

	//m = v - sT u
	short *m2_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
	memset(m2_ntt, 0, n*sizeof(short));
	for(int kx=0;kx<k;++kx)
		multiply_ntt_add(m2_ntt, pr_k.s_ntt+n*kx, u_ntt+n*kx, p);
		//for(int kv=0;kv<n;++kv)
		//	m2_ntt[kv]=kyber_montgomery_reduction(m2_ntt[kv]+pr_k.s_ntt[n*kx+kv]*u_ntt[n*kx+kv]);
	subtract_polynomials(m2_ntt, v_ntt, m2_ntt, n, q);
	//for(int kv=0;kv<n;++kv)
	//	m2_ntt[kv]=kyber_montgomery_reduction(v_ntt[kv]-m2_ntt[kv]);
	short *m2=(short*)_aligned_malloc(n*sizeof(short), align);
	apply_inverse_NTT(m2_ntt, m2, p);
//	std::cout<<"m2:", print_element(m2, n);//
	//m = compress(m, 1)
	memset(message2, 0, n>>3);
	for(int kv=0, q_4=q/4, q3_4=q*3/4;kv<n;++kv)
	{
		m2[kv]=(m2[kv]>q_4)&(m2[kv]<q3_4);
	//	m2[kv]/=q_2;
		message2[kv>>3]|=m2[kv]<<(kv&7);
	}
	
//	_aligned_free(ct.u), _aligned_free(ct.v);
	_aligned_free(u_ntt), _aligned_free(v_ntt);
	_aligned_free(m2_ntt), _aligned_free(m2);
}
void		kyber_encapsulate(Kyber_public_key const &pu_k, Kyber_KE_ciphertext &ke_ct, unsigned char *K, NTT_params const &p)
{
	const int align=sizeof(SIMD_type);
	const int size=32;

	//Encapsulate
	char m[size+1]={0};
	for(int k=0;k<size;++k)
		m[k]=(char)rand();
//	const char *m="12345678901234567890123456789012";//
//	std::cout<<"m:\n", print_buffer(m, size);//
	//G(pu_k, m) -> (K^, r, d) 3*256 bit
	int rho_size=size, t_size=1056, m_size=size,
		h_size=rho_size+t_size+m_size;
//	int h_size=32/2+256*3+32/2;
	unsigned char *h_buffer=new unsigned char[h_size];
	memcpy(h_buffer, pu_k.rho, rho_size*sizeof(unsigned char));
	memcpy(h_buffer+rho_size, pu_k.t, t_size*sizeof(unsigned char));
	memcpy(h_buffer+rho_size+t_size, m, m_size*sizeof(unsigned char));
	int Krd_size=size*3;
	unsigned char *Krd=new unsigned char[Krd_size];
//	std::cout<<"\npu_k, m:\n", print_buffer(h_buffer, h_size);//
	FIPS202_SHAKE128((unsigned char*)h_buffer, h_size, (unsigned char*)Krd, Krd_size);
	delete[] h_buffer;
//	std::cout<<"\nK^:\n", print_buffer(Krd, size);//
//	std::cout<<"r:\n", print_buffer(Krd+size, size);//
//	std::cout<<"d:\n", print_buffer(Krd+size*2, size);//
	
	Kyber_ciphertext ct;
//	std::cout<<"\npu_k.t:\n", print_buffer(pu_k.t, t_size);//
	kyber_encrypt(pu_k, m, Krd+size, ct, p);
//	kyber_encrypt(m, ct, pu_k, p);
//	std::cout<<"\npu_k.t:\n", print_buffer(pu_k.t, t_size);//
	//c = (u, v, d)
	ke_ct.u=(unsigned char*)ct.u, ke_ct.v=(unsigned char*)ct.v;
	ke_ct.d=(unsigned char*)_aligned_malloc(size*sizeof(unsigned char), align);
	memcpy(ke_ct.d, Krd+size*2, size);

	//K = H(K^, u, v, d)	16+256*3+[48]+16
	int u_size=1056, v_size=96,
		Kc_size=size+u_size+v_size+size;
	unsigned char *Kc=new unsigned char[Kc_size];
	memcpy(Kc, Krd, size);
	memcpy(Kc+size, ct.u, u_size);
	memcpy(Kc+size+u_size, ct.v, v_size);
	memcpy(Kc+size+u_size+v_size, Krd+64, size);
//	unsigned char *K=new unsigned char[size];
	FIPS202_SHAKE128(Kc, Kc_size, K, size);
	//std::cout<<"\nc = (u, v, d)\n";//
	//std::cout<<"u:\n", print_buffer(ct.u, u_size);//
	//std::cout<<"v:\n", print_buffer(ct.v, v_size);//
	//std::cout<<"d:\n", print_buffer(Krd+64, size);//
	//std::cout<<"K:\n", print_buffer(K, size);//
	delete[] Krd, Kc;
}
bool		kyber_decapsulate(Kyber_private_key const &pr_k, Kyber_public_key const &pu_k, Kyber_KE_ciphertext const &ke_ct, unsigned char *K2, NTT_params const &p)
{
	const int size=32;

	//Decapsulate
	unsigned char *m2=new unsigned char[size];
	Kyber_ciphertext ct;
	ct.u=(short*)ke_ct.u, ct.v=(short*)ke_ct.v;
	kyber_decrypt(ct, (char*)m2, pr_k, p);
//	std::cout<<"m2:\n", print_buffer(m2, size);//
	int rho_size=size, t_size=1056, m_size=size;
	int pum_size=rho_size+t_size+m_size;
	unsigned char *pum_buffer=new unsigned char[pum_size];
	memcpy(pum_buffer, pu_k.rho, rho_size);
	memcpy(pum_buffer+rho_size, pu_k.t, t_size);
	memcpy(pum_buffer+rho_size+t_size, m2, m_size);
	int Krd2_size=size*3;
	unsigned char *Krd2=new unsigned char[Krd2_size];
//	std::cout<<"\npu_k, m2:\n", print_buffer(pum_buffer, pum_size);//
	FIPS202_SHAKE128(pum_buffer, pum_size, Krd2, Krd2_size);
	delete[] pum_buffer;
//	std::cout<<"\nK2^:\n", print_buffer(Krd2, size);//
//	std::cout<<"r2:\n", print_buffer(Krd2+size, size);//
//	std::cout<<"d2:\n", print_buffer(Krd2+size*2, size);//
	Kyber_ciphertext ct2;
	kyber_encrypt(pu_k, (char*)m2, Krd2+size, ct2, p);
//	kyber_encrypt((char*)m2, ct2, pu_k, p);
	int error=0;
	int u_size=1056, v_size=96;
	for(int k=0, kEnd=u_size/sizeof(short);k<kEnd;++k)
		error|=ct.u[k]^ct2.u[k];
	for(int k=0, kEnd=v_size/sizeof(short);k<kEnd;++k)
		error|=ct.v[k]^ct2.v[k];
	for(int k=0;k<size;++k)
		error|=ke_ct.d[k]^Krd2[64+k];
//	std::cout<<"\nerror="<<error<<endl;
//	unsigned char *K2=new unsigned char[size];
	int Kc_size=size+u_size+v_size+size;
	unsigned char *Kc=new unsigned char[Kc_size];
	if(!error)
	{
		memcpy(Kc, Krd2, size);
		memcpy(Kc+size, ct.u, u_size);
		memcpy(Kc+size+u_size, ct.v, v_size);
		memcpy(Kc+size+u_size+v_size, Krd2+64, size);
	//	memcpy(K2, Krd2, size);
	}
	else
	{
		generate_uniform(size, Kc);
		memcpy(Kc+size, ct.u, u_size);
		memcpy(Kc+size+u_size, ct.v, v_size);
		memcpy(Kc+size+u_size+v_size, ke_ct.d, size);
	}
	FIPS202_SHAKE128(Kc, Kc_size, K2, size);
//	std::cout<<"\nK2:\n", print_buffer(K2, size);
//	_aligned_free(ct.u), _aligned_free(ct.v);
	delete[] m2, Krd2;
	return !error;
}

void pol_mul_sb(const short *a, const short *b, short *res, unsigned short p, unsigned n, unsigned start) //simple school book
{ // Polynomial multiplication using the schoolbook method, c[x] = a[x]*b[x] 
	// SECURITY NOTE: TO BE USED FOR TESTING ONLY.

	unsigned i, j,mask = 2*n - 1;
	
	short *c=(short *)malloc(mask*sizeof(short));
	//short c[2*n-1];
	for (i = 0; i < mask; i++) c[i] = 0;

	for (i = start; i < start+n; i++) {
		for (j = start; j < start+n; j++) {
			c[i+j-2*start]=c[i+j-2*start] + (a[i] * b[j]) & p-1;
			//c[i+j-2*start]=reduce(c[i+j-2*start] + (a[i] * b[j]), p);
			//printf("i+j : %u,i:%u,j:%u,a[i]=%lu,b[j]=%lu,c[i+j]=%lu\n",i+j,i,j,a[i],b[j],c[i+j]);
		}
	}
	for (i = 0; i < mask; i++){
		res[i] = (res[i]^res[i])+c[i];
		//res[i] = reduce(res[i]+c[i],p);
	}

	free(c);
}
void toom_cook_4way(const unsigned short *a1, const unsigned short *b1, unsigned short *result, unsigned long long p_mod, unsigned short n)
{
	const short SABER_N=256, small_len=SABER_N/4;
	short i;

	short result_final[2*SABER_N];
	unsigned long long p_mod_or=p_mod;
	p_mod<<=3;

//----------------array declaration to hold smaller arrays--------------------
	short a[small_len],b[small_len];
	short th_a[small_len],t_h_a[small_len];
	short th_b[small_len],t_h_b[small_len];
	short temp1[2*small_len];
//----------------array declaration to hold smaller arrays ends--------------------

//----------------array declaration to hold results--------------------
	short w1[2*small_len],w2[2*small_len],w3[2*small_len],w4[2*small_len],w5[2*small_len],w6[2*small_len],w7[2*small_len];
//----------------array declaration to hold results ends--------------------

	//--------------------these data are created for place holding---------
	short a1_ph[small_len],b1_ph[small_len];
	short a2_ph[small_len],b2_ph[small_len];
	short a3_ph[small_len],b3_ph[small_len];
	short a4_ph[small_len],b4_ph[small_len];
	short a5_ph[small_len],b5_ph[small_len];
	short a6_ph[small_len],b6_ph[small_len];
	//--------------------these data are created for place holding ends---------

	short inv3=-21845, inv9=-29127, inv15=-4369, int45=45, int30=30, int0=0;
	//short inv3=43691, inv9=36409, inv15=61167, int45=45, int30=30, int0=0;

	//do the partial products
	//-------------------t0--------------------		w7
	//create a(0)*b(0)
	for(i=0;i<small_len;i++){
		a1_ph[i]=a1[i+0];
		b1_ph[i]=b1[i+0];
	}
	//-------------------t0 ends------------------

	//-------------------th and t_h. th <-a(1/2)*b(1/2). t_h <- a(-1/2)*b(-1/2) ---------------------		w5, w6
	//create partial sum for th and t_h
	for(i=0;i<small_len;i++)
	{
		th_a[i]= a1_ph[i]<<2;//th_x contains 4*x[0]
		th_b[i]= b1_ph[i]<<2;
		
		th_a[i]= th_a[i]+a1[small_len*2+i];//th_x contains 4*x[0]+x[2]
		th_b[i]= th_b[i]+b1[small_len*2+i];
		
		th_a[i]= th_a[i]<<1;//th_x_avx contains 8*x[0]+2*x[2]
		th_b[i]= th_b[i]<<1;
		
		t_h_a[i]= a1[small_len*1+i];//t_h_x_avx contains x[1]
		t_h_b[i]= b1[small_len*1+i];
		
		t_h_a[i]= t_h_a[i]<<2;//t_h_x_avx contains 4*x[1]
		t_h_b[i]= t_h_b[i]<<2;

		t_h_a[i]= t_h_a[i]+a1[small_len*3+i];//th_x_avx contains 4*x[1]+x[3]
		t_h_b[i]= t_h_b[i]+b1[small_len*3+i];
	}

	//create th
	for(i=0;i<small_len;i++){
		a2_ph[i]= th_a[i]+t_h_a[i];
		b2_ph[i]= th_b[i]+t_h_b[i];
	}
	//pol_mul_avx(a_avx, b_avx, w5_avx, p_mod, small_len, 0);	

	//create t_h
	for(i=0;i<small_len;i++){
		a3_ph[i]= th_a[i]-t_h_a[i];
		b3_ph[i]= th_b[i]-t_h_b[i];
	}
	//pol_mul_avx(a_avx, b_avx, w6_avx, p_mod, small_len, 0);

	//-------------------t1 and t_1. t1 <-a(1)*b(1). t_1 <- a(-1)*b(-1) ---------------------		w3, w4

	for(i=0;i<small_len;i++)//create partial sum for t_1 and t1
	{
		th_a[i]= a1[small_len*2+i]+a1[small_len*0+i];//th_x_avx contains x[2]+x[0]
		th_b[i]= b1[small_len*2+i]+b1[small_len*0+i];
		
		t_h_a[i]= a1[small_len*3+i]+a1[small_len*1+i];//th_x_avx contains x[3]+x[1]
		t_h_b[i]= b1[small_len*3+i]+b1[small_len*1+i];
	}

	//create t1
	for(i=0;i<small_len;i++){
		a4_ph[i]= th_a[i]+t_h_a[i];// x[0]+x[1]+x[2]+x[3]
		b4_ph[i]= th_b[i]+t_h_b[i];
	}
	//pol_mul_avx(a_avx, b_avx, w3_avx, p_mod, small_len, 0);	

	//create t_1
	for(i=0;i<small_len;i++){
		a5_ph[i]= th_a[i]-t_h_a[i];//-x[3]+x[2]-x[1]+x[0]
		b5_ph[i]= th_b[i]-t_h_b[i];
	}
	//pol_mul_avx(a_avx, b_avx, w4_avx, p_mod, small_len, 0);	

	//------------------t_inf------------------------------			w1
	//create t_inf
	for(i=0;i<small_len;i++){
		a6_ph[i]= a1[small_len*3+i];//x_avx contains x[3]
		b6_ph[i]= b1[small_len*3+i];
	}
	//pol_mul_avx(a_avx, b_avx, w1_avx, p_mod, small_len, 0);

	//-------------------t_inf ends----------------------
	
	//-------------------t2-------------------------			w2
	for(i=0;i<small_len;i++)
	{
		a[i]= a6_ph[i]+a1[small_len*3+i];// 2*x[3]
		b[i]= b6_ph[i]+b1[small_len*3+i];

		a[i]= a[i]+a1[small_len*2+i];// 2*x[3]+x[2]
		b[i]= b[i]+b1[small_len*2+i];
		
		a[i]= a[i]<<1;// 4*x[3]+2*x[2]
		b[i]= b[i]<<1;
		
		a[i]= a[i]+a1[small_len*1+i];// 4*x[3]+2*x[2]+x[1]
		b[i]= b[i]+b1[small_len*1+i];
		
		a[i]= a[i]<<1;// 8*x[3]+4*x[2]+2*x[1]
		b[i]= b[i]<<1;
		
		a[i]= a[i]+a1[small_len*0+i];// 8*x[3]+8*x[2]+2*x[1]+x[0]
		b[i]= b[i]+b1[small_len*0+i];
	}
	
	pol_mul_sb(a1_ph, b1_ph, w7, (short)p_mod, small_len, 0);//-----first multiplication
	pol_mul_sb(a2_ph, b2_ph, w5, (short)p_mod, small_len, 0);//-----second multiplication
	pol_mul_sb(a3_ph, b3_ph, w6, (short)p_mod, small_len, 0);//-----Third multiplication
	pol_mul_sb(a4_ph, b4_ph, w3, (short)p_mod, small_len, 0);//-----Fourth multilication
	pol_mul_sb(a5_ph, b5_ph, w4, (short)p_mod, small_len, 0);//-----Fifth Multiplication
	pol_mul_sb(a6_ph, b6_ph, w1, (short)p_mod, small_len, 0);//-----Sixth Multiplication
	pol_mul_sb(a,	  b,	 w2, (short)p_mod, small_len, 0);//-----Seventh Multiplication
	//std::cout<<"w1:", print_element(w1, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w2:", print_element(w2, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w3:", print_element(w3, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w4:", print_element(w4, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w5:", print_element(w5, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w6:", print_element(w6, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w7:", print_element(w7, 2*small_len, (int)p_mod_or);//

	//	--------------------------------------------
	//	---------------Solution starts--------------
	//	--------------------------------------------
	for(i=0;i<2*small_len;i++)
	{
		w2[i]= w2[i]+w5[i];//w2 <- w2+w5
		w6[i]= w6[i]-w5[i];// w6 <- w6-w5
		w4[i]= w4[i]-w3[i];// w4 <- w4-w3
		
		w5[i]= w5[i]-w1[i];// w5 <- w5-w1
		temp1[i] = w7[i]<<6; //temp <- 64*w7
		w5[i]= w5[i]-temp1[i];// w5 <- w5-64*w7

		w4[i] = w4[i]>>1; //w4 <- w4/2
		w3[i] = w3[i]+w4[i];//w3 <- w3+w4

		temp1[i] = w5[i]<<1; //temp <- 2*w5
		w5[i]= w6[i]+temp1[i];//w5 <- 2*w5+w6

		temp1[i] = w3[i]<<6; //temp <- 64*w3
		temp1[i] = w3[i]+temp1[i]; //temp <- 65*w3
		w2[i]= w2[i]-temp1[i];// w2 <- w2-65*w3

		w3[i]= w3[i]-w7[i];// w3 <- w3-w7
		w3[i]= w3[i]-w1[i];// w3 <- w3-w1

		temp1[i] = w3[i]*int45; //temp <- 45*w3
		w2[i] = w2[i]+temp1[i]; //w2 <- w2+45*w3

		temp1[i] = w3[i]<<3; //temp <- 8*w3
		w5[i]= w5[i]-temp1[i];//w5 <- w5-8*w3
		w5[i] = w5[i]*inv3; //w5 <- w5*1/3
		w5[i] = w5[i]>>3; //w5 <- w5*1/8 ---> w5=w5/24

		w6[i] = w2[i]+w6[i]; //w6 <- w6+w2

		temp1[i] = w4[i]<<4; //temp <- 16*w4
		w2[i] = w2[i]+temp1[i]; //w2 <- w2+16*w4
		w2[i] = w2[i]*inv9; //w2 <- w2*1/9
		w2[i] = w2[i]>>1; //w2 <- w2*1/2 ---> w2=w2/18

		w3[i]= w3[i]-w5[i];//w3 <- w3-w5
		
		w4[i] = w4[i]+w2[i]; //w4 <- w4+w2

		w4[i] = int0-w4[i]; //w4 <- -(w4+w2)

		temp1[i] = w2[i]*int30; //temp <- w2*30
		w6[i]= temp1[i]-w6[i];//w6 <- 30*w2-w6
		w6[i] = w6[i]*inv15; //w6 <- w6*1/15
		w6[i] = w6[i]>>2; //w6 <- w6*1/4 ---> w6=w6/60

		w2[i]= w2[i]-w6[i];//w2 <- w2-w6
	}
	//std::cout<<"w1:", print_element(w1, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w2:", print_element(w2, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w3:", print_element(w3, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w4:", print_element(w4, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w5:", print_element(w5, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w6:", print_element(w6, 2*small_len, (int)p_mod_or);//
	//std::cout<<"w7:", print_element(w7, 2*small_len, (int)p_mod_or);//

	for(i=0; i<2*SABER_N; i++)
		result_final[i] = 0;

	for(i=0;i<2*small_len-1;i++)
	{	
		result_final[0*small_len+i]= result_final[0*small_len+i]+ w7[i];
		result_final[1*small_len+i]= result_final[1*small_len+i]+ w6[i];
		result_final[2*small_len+i]= result_final[2*small_len+i]+ w5[i];
		result_final[3*small_len+i]= result_final[3*small_len+i]+ w4[i];
		result_final[4*small_len+i]= result_final[4*small_len+i]+ w3[i];
		result_final[5*small_len+i]= result_final[5*small_len+i]+ w2[i];
		result_final[6*small_len+i]= result_final[6*small_len+i]+ w1[i];
	}		
	
	//---------------reduction-------
	for(i=n;i<2*n-1;i++)
		result_final[i-n]=result_final[i-n]-result_final[i];
	result_final[n]=0; //256th coefficient=0;

	//----------------------copy result back----------------
	for(i=0; i<n; i++)
		result[i] = result[i]+result_final[i] & p_mod_or-1;
	//	result[i] = result_final[i] & p_mod_or-1;
	//	result[i] = reduce(result_final[i], );
}//*/
void		multiply_polynomials_sb(const short *a, const short *b, short *result, unsigned n) //simple school book
{
	unsigned result_size=2*n-1;
	memset(result, 0, result_size*sizeof(short));
	for(unsigned i=0;i<n;i++)
		for(unsigned j=0;j<n;j++)
			result[i+j]+=a[i]*b[j];
}
void		multiply_polynomials_sb_2(const short *a, const short *b, short *ab)
{
	ab[0]=a[0]*b[0];
	ab[1]=a[1]*b[0]+a[0]*b[1];//0.905ms 1175168c
	ab[2]=			a[1]*b[1];
}
void		multiply_polynomials_sb_4(const short *a, const short *b, short *ab)
{
	ab[0]=a[0]*b[0];
	ab[1]=a[1]*b[0]+a[0]*b[1];
	ab[2]=a[2]*b[0]+a[1]*b[1]+a[0]*b[2];
	ab[3]=a[3]*b[0]+a[2]*b[1]+a[1]*b[2]+a[0]*b[3];//0.409ms 531498c
	ab[4]=			a[3]*b[1]+a[2]*b[2]+a[1]*b[3];
	ab[5]=					  a[3]*b[2]+a[2]*b[3];
	ab[6]=								a[3]*b[3];
}
void		multiply_polynomials_sb_8(const short *a, const short *b, short *ab)
{
#if PROCESSOR_ARCH>=SSE2
	__m128i a0=_mm_loadu_si128((__m128i*)a);
	__m128i p0=_mm_mullo_epi16(a0, _mm_set1_epi16(b[0]));//0.19ms 255696c
	__m128i p1=_mm_mullo_epi16(a0, _mm_set1_epi16(b[1]));
	__m128i p2=_mm_mullo_epi16(a0, _mm_set1_epi16(b[2]));
	__m128i p3=_mm_mullo_epi16(a0, _mm_set1_epi16(b[3]));
	__m128i p4=_mm_mullo_epi16(a0, _mm_set1_epi16(b[4]));
	__m128i p5=_mm_mullo_epi16(a0, _mm_set1_epi16(b[5]));
	__m128i p6=_mm_mullo_epi16(a0, _mm_set1_epi16(b[6]));
	__m128i p7=_mm_mullo_epi16(a0, _mm_set1_epi16(b[7]));

	__m128i v_lo=_mm_add_epi16(p0, _mm_slli_si128(p1, 2));
	v_lo=_mm_add_epi16(v_lo, _mm_slli_si128(p2, 4));
	v_lo=_mm_add_epi16(v_lo, _mm_slli_si128(p3, 6));
	v_lo=_mm_add_epi16(v_lo, _mm_slli_si128(p4, 8));
	v_lo=_mm_add_epi16(v_lo, _mm_slli_si128(p5, 10));
	v_lo=_mm_add_epi16(v_lo, _mm_slli_si128(p6, 12));
	v_lo=_mm_add_epi16(v_lo, _mm_slli_si128(p7, 14));

	__m128i v_hi=_mm_add_epi16(p7, _mm_srli_si128(p6, 2));
	v_hi=_mm_add_epi16(v_hi, _mm_srli_si128(p5, 4));
	v_hi=_mm_add_epi16(v_hi, _mm_srli_si128(p4, 6));
	v_hi=_mm_add_epi16(v_hi, _mm_srli_si128(p3, 8));
	v_hi=_mm_add_epi16(v_hi, _mm_srli_si128(p2, 10));
	v_hi=_mm_add_epi16(v_hi, _mm_srli_si128(p1, 12));
	v_hi=_mm_add_epi16(v_hi, _mm_srli_si128(p0, 14));
	_mm_storeu_si128((__m128i*)ab, v_lo);
	_mm_storeu_si128((__m128i*)(ab+7), v_hi);
#else//*/
	ab[0]=a[0]*b[0];
	ab[1]=a[1]*b[0]+a[0]*b[1];
	ab[2]=a[2]*b[0]+a[1]*b[1]+a[0]*b[2];
	ab[3]=a[3]*b[0]+a[2]*b[1]+a[1]*b[2]+a[0]*b[3];
	ab[4]=a[4]*b[0]+a[3]*b[1]+a[2]*b[2]+a[1]*b[3]+a[0]*b[4];
	ab[5]=a[5]*b[0]+a[4]*b[1]+a[3]*b[2]+a[2]*b[3]+a[1]*b[4]+a[0]*b[5];
	ab[6]=a[6]*b[0]+a[5]*b[1]+a[4]*b[2]+a[3]*b[3]+a[2]*b[4]+a[1]*b[5]+a[0]*b[6];
	ab[7]=a[7]*b[0]+a[6]*b[1]+a[5]*b[2]+a[4]*b[3]+a[3]*b[4]+a[2]*b[5]+a[1]*b[6]+a[0]*b[7];//0.237ms 307957c
	ab[8]=			a[7]*b[1]+a[6]*b[2]+a[5]*b[3]+a[4]*b[4]+a[3]*b[5]+a[2]*b[6]+a[1]*b[7];
	ab[9]=					  a[7]*b[2]+a[6]*b[3]+a[5]*b[4]+a[4]*b[5]+a[3]*b[6]+a[2]*b[7];
	ab[10]=								a[7]*b[3]+a[6]*b[4]+a[5]*b[5]+a[4]*b[6]+a[3]*b[7];
	ab[11]=										  a[7]*b[4]+a[6]*b[5]+a[5]*b[6]+a[4]*b[7];
	ab[12]=													a[7]*b[5]+a[6]*b[6]+a[5]*b[7];
	ab[13]=															  a[7]*b[6]+a[6]*b[7];
	ab[14]=																		a[7]*b[7];
#endif
}
void		multiply_polynomials_sb_16(const short *a, const short *b, short *ab)
{
#if PROCESSOR_ARCH>=AVX2
	__m256i a0=_mm256_loadu_si256((__m256i*)a);
	__m256i p0=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[0]));
	__m256i p1=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[1]));
	__m256i p2=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[2]));
	__m256i p3=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[3]));
	__m256i p4=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[4]));
	__m256i p5=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[5]));
	__m256i p6=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[6]));
	__m256i p7=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[7]));
	__m256i p8=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[8]));
	__m256i p9=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[9]));
	__m256i p10=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[10]));
	__m256i p11=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[11]));
	__m256i p12=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[12]));
	__m256i p13=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[13]));
	__m256i p14=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[14]));
	__m256i p15=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[15]));
	
//	auto LOL_1=_mm256_alignr_epi8(p1, _mm256_permute2x128_si256(p1, p1, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 2);
	__m256i v_lo=_mm256_add_epi16(p0, _mm256_alignr_epi8(p1, _mm256_permute2x128_si256(p1, p1, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 2));//alignr({Hi, LO}, {LO, 0}, 16-N)
	v_lo=_mm256_add_epi16(v_lo, _mm256_alignr_epi8(p2, _mm256_permute2x128_si256(p2, p2, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 4));
	v_lo=_mm256_add_epi16(v_lo, _mm256_alignr_epi8(p3, _mm256_permute2x128_si256(p3, p3, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 6));
	v_lo=_mm256_add_epi16(v_lo, _mm256_alignr_epi8(p4, _mm256_permute2x128_si256(p4, p4, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 8));
	v_lo=_mm256_add_epi16(v_lo, _mm256_alignr_epi8(p5, _mm256_permute2x128_si256(p5, p5, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 10));
	v_lo=_mm256_add_epi16(v_lo, _mm256_alignr_epi8(p6, _mm256_permute2x128_si256(p6, p6, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 12));
	v_lo=_mm256_add_epi16(v_lo, _mm256_alignr_epi8(p7, _mm256_permute2x128_si256(p7, p7, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 14));
	v_lo=_mm256_add_epi16(v_lo, _mm256_permute2x128_si256(p8, p8, _MM_SHUFFLE(0, 0, 2, 0)));
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p9, p9, _MM_SHUFFLE(0, 0, 2, 0)), 18 - 16));//slli({LO, 0})
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p10, p10, _MM_SHUFFLE(0, 0, 2, 0)), 20 - 16));
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p11, p11, _MM_SHUFFLE(0, 0, 2, 0)), 22 - 16));
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p12, p12, _MM_SHUFFLE(0, 0, 2, 0)), 24 - 16));
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p13, p13, _MM_SHUFFLE(0, 0, 2, 0)), 26 - 16));
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p14, p14, _MM_SHUFFLE(0, 0, 2, 0)), 28 - 16));
	v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(_mm256_permute2x128_si256(p15, p15, _MM_SHUFFLE(0, 0, 2, 0)), 30 - 16));
	
//	auto LOL_2=_mm256_alignr_epi8(_mm256_permute2x128_si256(p14, p14, _MM_SHUFFLE(2, 0, 0, 1)), p14, 2);//
	__m256i v_hi=_mm256_add_epi16(p15, _mm256_alignr_epi8(_mm256_permute2x128_si256(p14, p14, _MM_SHUFFLE(2, 0, 0, 1)), p14, 2));//alignr({0, HI}, {HI, LO}, N)
	v_hi=_mm256_add_epi16(v_hi, _mm256_alignr_epi8(_mm256_permute2x128_si256(p13, p13, _MM_SHUFFLE(2, 0, 0, 1)), p13, 4));
	v_hi=_mm256_add_epi16(v_hi, _mm256_alignr_epi8(_mm256_permute2x128_si256(p12, p12, _MM_SHUFFLE(2, 0, 0, 1)), p12, 6));
	v_hi=_mm256_add_epi16(v_hi, _mm256_alignr_epi8(_mm256_permute2x128_si256(p11, p11, _MM_SHUFFLE(2, 0, 0, 1)), p11, 8));
	v_hi=_mm256_add_epi16(v_hi, _mm256_alignr_epi8(_mm256_permute2x128_si256(p10, p10, _MM_SHUFFLE(2, 0, 0, 1)), p10, 10));
	v_hi=_mm256_add_epi16(v_hi, _mm256_alignr_epi8(_mm256_permute2x128_si256(p9, p9, _MM_SHUFFLE(2, 0, 0, 1)), p9, 12));
	v_hi=_mm256_add_epi16(v_hi, _mm256_alignr_epi8(_mm256_permute2x128_si256(p8, p8, _MM_SHUFFLE(2, 0, 0, 1)), p8, 14));
	v_hi=_mm256_add_epi16(v_hi, _mm256_permute2x128_si256(p7, p7, _MM_SHUFFLE(2, 0, 0, 1)));
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p6, p6, _MM_SHUFFLE(2, 0, 0, 1)), 18 - 16));//srli({0, HI}, N-16)
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p5, p5, _MM_SHUFFLE(2, 0, 0, 1)), 20 - 16));
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p4, p4, _MM_SHUFFLE(2, 0, 0, 1)), 22 - 16));
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p3, p3, _MM_SHUFFLE(2, 0, 0, 1)), 24 - 16));
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p2, p2, _MM_SHUFFLE(2, 0, 0, 1)), 26 - 16));
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p1, p1, _MM_SHUFFLE(2, 0, 0, 1)), 28 - 16));
	v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(_mm256_permute2x128_si256(p0, p0, _MM_SHUFFLE(2, 0, 0, 1)), 30 - 16));
	_mm256_storeu_si256((__m256i*)ab, v_lo);
	_mm256_storeu_si256((__m256i*)(ab+15), v_hi);


	//const __m256i shr02=_mm256_set_epi8(32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2);
	//const __m256i shr04=_mm256_set_epi8(32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
	//const __m256i shr06=_mm256_set_epi8(32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6);
	//const __m256i shr08=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8);
	//const __m256i shr10=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10);
	//const __m256i shr12=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12);
	//const __m256i shr14=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);
	//const __m256i shr16=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);
	//const __m256i shr18=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18);
	//const __m256i shr20=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20);
	//const __m256i shr22=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24, 23, 22);
	//const __m256i shr24=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26, 25, 24);
	//const __m256i shr26=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28, 27, 26);
	//const __m256i shr28=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30, 29, 28);
	//const __m256i shr30=_mm256_set_epi8(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,		31, 30);
	//
	//const __m256i shl02=_mm256_set_epi8(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32);
	//const __m256i shl04=_mm256_set_epi8(27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32);
	//const __m256i shl06=_mm256_set_epi8(25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32);
	//const __m256i shl08=_mm256_set_epi8(23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl10=_mm256_set_epi8(21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl12=_mm256_set_epi8(19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl14=_mm256_set_epi8(17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl16=_mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl18=_mm256_set_epi8(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl20=_mm256_set_epi8(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl22=_mm256_set_epi8(9, 8, 7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl24=_mm256_set_epi8(7, 6, 5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl26=_mm256_set_epi8(5, 4, 3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl28=_mm256_set_epi8(3, 2, 1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	//const __m256i shl30=_mm256_set_epi8(1, 0,		32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32);
	////const __m256i idx=_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	//__m256i a0=_mm256_loadu_si256((__m256i*)a);
	//__m256i p0=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[0]));
	//__m256i p1=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[1]));
	//__m256i p2=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[2]));
	//__m256i p3=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[3]));
	//__m256i p4=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[4]));
	//__m256i p5=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[5]));
	//__m256i p6=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[6]));
	//__m256i p7=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[7]));
	//__m256i p8=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[8]));
	//__m256i p9=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[9]));
	//__m256i p10=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[10]));
	//__m256i p11=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[11]));
	//__m256i p12=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[12]));
	//__m256i p13=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[13]));
	//__m256i p14=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[14]));
	//__m256i p15=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[15]));
	//
	//auto LOL_1=_mm256_permutevar8x32_epi32(p1, shl02);
	//__m256i v_lo=_mm256_add_epi16(p0, _mm256_permutevar8x32_epi32(p1, shl02));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p2, shl04));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p3, shl06));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p4, shl08));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p5, shl10));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p6, shl12));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p7, shl14));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p8, shl16));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p9, shl18));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p10, shl20));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p11, shl22));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p12, shl24));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p13, shl26));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p14, shl28));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_permutevar8x32_epi32(p15, shl30));
	//
	//__m256i v_hi=_mm256_add_epi16(p15, _mm256_permutevar8x32_epi32(p14, shr02));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p13, shr04));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p12, shr06));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p11, shr08));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p10, shr10));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p9, shr12));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p8, shr14));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p7, shr16));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p6, shr18));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p5, shr20));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p4, shr22));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p3, shr24));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p2, shr26));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p1, shr28));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_permutevar8x32_epi32(p0, shr30));
	//_mm256_storeu_si256((__m256i*)ab, v_lo);
	//_mm256_storeu_si256((__m256i*)(ab+15), v_hi);


	//__m256i a0=_mm256_loadu_si256((__m256i*)a);
	//__m256i p0=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[0]));
	//__m256i p1=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[1]));
	//__m256i p2=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[2]));
	//__m256i p3=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[3]));
	//__m256i p4=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[4]));
	//__m256i p5=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[5]));
	//__m256i p6=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[6]));
	//__m256i p7=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[7]));
	//__m256i p8=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[8]));
	//__m256i p9=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[9]));
	//__m256i p10=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[10]));
	//__m256i p11=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[11]));
	//__m256i p12=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[12]));
	//__m256i p13=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[13]));
	//__m256i p14=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[14]));
	//__m256i p15=_mm256_mullo_epi16(a0, _mm256_set1_epi16(b[15]));
	//
	//__m256i v_lo=_mm256_add_epi16(p0, _mm256_slli_si256(p1, 2));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p2, 4));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p3, 6));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p4, 8));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p5, 10));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p6, 12));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p7, 14));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p8, 16));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p9, 18));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p10, 20));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p11, 22));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p12, 24));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p13, 26));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p14, 28));
	//v_lo=_mm256_add_epi16(v_lo, _mm256_slli_si256(p15, 30));
	//
	//__m256i v_hi=_mm256_add_epi16(p15, _mm256_srli_si256(p14, 2));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p13, 4));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p12, 6));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p11, 8));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p10, 10));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p9, 12));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p8, 14));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p7, 16));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p6, 18));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p5, 20));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p4, 22));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p3, 24));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p2, 26));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p1, 28));
	//v_hi=_mm256_add_epi16(v_hi, _mm256_srli_si256(p0, 30));
	//_mm256_storeu_si256((__m256i*)ab, v_lo);
	//_mm256_storeu_si256((__m256i*)(ab+15), v_hi);
#else//*/
	ab[ 0]=a[ 0]*b[0];
	ab[ 1]=a[ 1]*b[0] +	a[ 0]*b[1];
	ab[ 2]=a[ 2]*b[0] +	a[ 1]*b[1] + a[ 0]*b[2];
	ab[ 3]=a[ 3]*b[0] +	a[ 2]*b[1] + a[ 1]*b[2] + a[ 0]*b[3];
	ab[ 4]=a[ 4]*b[0] +	a[ 3]*b[1] + a[ 2]*b[2] + a[ 1]*b[3] + a[ 0]*b[4];
	ab[ 5]=a[ 5]*b[0] +	a[ 4]*b[1] + a[ 3]*b[2] + a[ 2]*b[3] + a[ 1]*b[4] + a[ 0]*b[5];
	ab[ 6]=a[ 6]*b[0] +	a[ 5]*b[1] + a[ 4]*b[2] + a[ 3]*b[3] + a[ 2]*b[4] + a[ 1]*b[5] + a[ 0]*b[6];
	ab[ 7]=a[ 7]*b[0] +	a[ 6]*b[1] + a[ 5]*b[2] + a[ 4]*b[3] + a[ 3]*b[4] + a[ 2]*b[5] + a[ 1]*b[6] + a[ 0]*b[7];
	ab[ 8]=a[ 8]*b[0] +	a[ 7]*b[1] + a[ 6]*b[2] + a[ 5]*b[3] + a[ 4]*b[4] + a[ 3]*b[5] + a[ 2]*b[6] + a[ 1]*b[7] + a[ 0]*b[8];
	ab[ 9]=a[ 9]*b[0] +	a[ 8]*b[1] + a[ 7]*b[2] + a[ 6]*b[3] + a[ 5]*b[4] + a[ 4]*b[5] + a[ 3]*b[6] + a[ 2]*b[7] + a[ 1]*b[8] + a[ 0]*b[9];
	ab[10]=a[10]*b[0] +	a[ 9]*b[1] + a[ 8]*b[2] + a[ 7]*b[3] + a[ 6]*b[4] + a[ 5]*b[5] + a[ 4]*b[6] + a[ 3]*b[7] + a[ 2]*b[8] + a[ 1]*b[9] + a[ 0]*b[10];
	ab[11]=a[11]*b[0] +	a[10]*b[1] + a[ 9]*b[2] + a[ 8]*b[3] + a[ 7]*b[4] + a[ 6]*b[5] + a[ 5]*b[6] + a[ 4]*b[7] + a[ 3]*b[8] + a[ 2]*b[9] + a[ 1]*b[10] + a[ 0]*b[11];
	ab[12]=a[12]*b[0] +	a[11]*b[1] + a[10]*b[2] + a[ 9]*b[3] + a[ 8]*b[4] + a[ 7]*b[5] + a[ 6]*b[6] + a[ 5]*b[7] + a[ 4]*b[8] + a[ 3]*b[9] + a[ 2]*b[10] + a[ 1]*b[11] + a[ 0]*b[12];
	ab[13]=a[13]*b[0] +	a[12]*b[1] + a[11]*b[2] + a[10]*b[3] + a[ 9]*b[4] + a[ 8]*b[5] + a[ 7]*b[6] + a[ 6]*b[7] + a[ 5]*b[8] + a[ 4]*b[9] + a[ 3]*b[10] + a[ 2]*b[11] + a[ 1]*b[12] + a[ 0]*b[13];
	ab[14]=a[14]*b[0] +	a[13]*b[1] + a[12]*b[2] + a[11]*b[3] + a[10]*b[4] + a[ 9]*b[5] + a[ 8]*b[6] + a[ 7]*b[7] + a[ 6]*b[8] + a[ 5]*b[9] + a[ 4]*b[10] + a[ 3]*b[11] + a[ 2]*b[12] + a[ 1]*b[13] + a[ 0]*b[14];
	ab[15]=a[15]*b[0] +	a[14]*b[1] + a[13]*b[2] + a[12]*b[3] + a[11]*b[4] + a[10]*b[5] + a[ 9]*b[6] + a[ 8]*b[7] + a[ 7]*b[8] + a[ 6]*b[9] + a[ 5]*b[10] + a[ 4]*b[11] + a[ 3]*b[12] + a[ 2]*b[13] + a[ 1]*b[14] + a[ 0]*b[15];
	ab[16]=				a[15]*b[1] + a[14]*b[2] + a[13]*b[3] + a[12]*b[4] + a[11]*b[5] + a[10]*b[6] + a[ 9]*b[7] + a[ 8]*b[8] + a[ 7]*b[9] + a[ 6]*b[10] + a[ 5]*b[11] + a[ 4]*b[12] + a[ 3]*b[13] + a[ 2]*b[14] + a[ 1]*b[15];
	ab[17]=							 a[15]*b[2] + a[14]*b[3] + a[13]*b[4] + a[12]*b[5] + a[11]*b[6] + a[10]*b[7] + a[ 9]*b[8] + a[ 8]*b[9] + a[ 7]*b[10] + a[ 6]*b[11] + a[ 5]*b[12] + a[ 4]*b[13] + a[ 3]*b[14] + a[ 2]*b[15];
	ab[18]=										  a[15]*b[3] + a[14]*b[4] + a[13]*b[5] + a[12]*b[6] + a[11]*b[7] + a[10]*b[8] + a[ 9]*b[9] + a[ 8]*b[10] + a[ 7]*b[11] + a[ 6]*b[12] + a[ 5]*b[13] + a[ 4]*b[14] + a[ 3]*b[15];
	ab[19]=													   a[15]*b[4] + a[14]*b[5] + a[13]*b[6] + a[12]*b[7] + a[11]*b[8] + a[10]*b[9] + a[ 9]*b[10] + a[ 8]*b[11] + a[ 7]*b[12] + a[ 6]*b[13] + a[ 5]*b[14] + a[ 4]*b[15];
	ab[20]=																	a[15]*b[5] + a[14]*b[6] + a[13]*b[7] + a[12]*b[8] + a[11]*b[9] + a[10]*b[10] + a[ 9]*b[11] + a[ 8]*b[12] + a[ 7]*b[13] + a[ 6]*b[14] + a[ 5]*b[15];
	ab[21]=																				 a[15]*b[6] + a[14]*b[7] + a[13]*b[8] + a[12]*b[9] + a[11]*b[10] + a[10]*b[11] + a[ 9]*b[12] + a[ 8]*b[13] + a[ 7]*b[14] + a[ 6]*b[15];
	ab[22]=																							  a[15]*b[7] + a[14]*b[8] + a[13]*b[9] + a[12]*b[10] + a[11]*b[11] + a[10]*b[12] + a[ 9]*b[13] + a[ 8]*b[14] + a[ 7]*b[15];
	ab[23]=																										   a[15]*b[8] + a[14]*b[9] + a[13]*b[10] + a[12]*b[11] + a[11]*b[12] + a[10]*b[13] + a[ 9]*b[14] + a[ 8]*b[15];
	ab[24]=																														a[15]*b[9] + a[14]*b[10] + a[13]*b[11] + a[12]*b[12] + a[11]*b[13] + a[10]*b[14] + a[ 9]*b[15];
	ab[25]=																																	 a[15]*b[10] + a[14]*b[11] + a[13]*b[12] + a[12]*b[13] + a[11]*b[14] + a[10]*b[15];
	ab[26]=																																				   a[15]*b[11] + a[14]*b[12] + a[13]*b[13] + a[12]*b[14] + a[11]*b[15];
	ab[27]=																																								 a[15]*b[12] + a[14]*b[13] + a[13]*b[14] + a[12]*b[15];
	ab[28]=																																											   a[15]*b[13] + a[14]*b[14] + a[13]*b[15];
	ab[29]=																																															 a[15]*b[14] + a[14]*b[15];
	ab[30]=																																																		   a[15]*b[15];
#endif
	//std::cout<<"a:\t", print_element(a, 16, 1<<13);//
	//std::cout<<"b:\t", print_element(b, 16, 1<<13);//
	//std::cout<<"ab:\t", print_element(ab, 31, 1<<13);//
}
void		multiply_karatsuba_noreduction(const short *a, const short *b, short *ab, int n)
{
	const int n_2=n>>1, sub_size=n-1, buf_size=sub_size*5;
	const short *A0=a, *A1=a+n_2, *B0=b, *B1=b+n_2;
	short *buffer=new short[buf_size],
		*C0=buffer, *C1=buffer+sub_size, *C2=buffer+sub_size*2,
		*t1=buffer+sub_size*3, *t2=buffer+sub_size*4;
	memset(buffer, 0, buf_size*sizeof(short));
#if PROCESSOR_ARCH>=AVX2
	if(n_2>=16)
		for(int k=0;k<n_2;k+=16)
		{
			__m256i a0=_mm256_loadu_si256((__m256i*)(A0+k));
			__m256i a1=_mm256_loadu_si256((__m256i*)(A1+k));
			__m256i b0=_mm256_loadu_si256((__m256i*)(B0+k));
			__m256i b1=_mm256_loadu_si256((__m256i*)(B1+k));
			__m256i v1=_mm256_add_epi16(a0, a1);
			__m256i v2=_mm256_add_epi16(b0, b1);
			_mm256_storeu_si256((__m256i*)(t1+k), v1);
			_mm256_storeu_si256((__m256i*)(t2+k), v2);
		}
	else
#elif PROCESSOR_ARCH>=SSE2
	if(n_2>=8)
		for(int k=0;k<n_2;k+=8)
		{
			__m128i a0=_mm_loadu_si128((__m128i*)(A0+k));
			__m128i a1=_mm_loadu_si128((__m128i*)(A1+k));
			__m128i b0=_mm_loadu_si128((__m128i*)(B0+k));
			__m128i b1=_mm_loadu_si128((__m128i*)(B1+k));
			__m128i v1=_mm_add_epi16(a0, a1);
			__m128i v2=_mm_add_epi16(b0, b1);
			_mm_storeu_si128((__m128i*)(t1+k), v1);
			_mm_storeu_si128((__m128i*)(t2+k), v2);
		}
	else
#endif//*/
		for(int k=0;k<n_2;++k)
		{
			t1[k]=A0[k]+A1[k];
			t2[k]=B0[k]+B1[k];
		}
	if(n_2>16)
//	if(n_2>8)
//	if(n_2>4)
//	if(n_2>2)
	{
		multiply_karatsuba_noreduction(A0, B0, C0, n_2);
		multiply_karatsuba_noreduction(t1, t2, C1, n_2);
		multiply_karatsuba_noreduction(A1, B1, C2, n_2);
	}
	else//if(n_2==16)
	{
		multiply_polynomials_sb_16(A0, B0, C0);
		multiply_polynomials_sb_16(t1, t2, C1);
		multiply_polynomials_sb_16(A1, B1, C2);
	}
	//else//if(n_2==8)
	//{
	//	multiply_polynomials_sb_8(A0, B0, C0);
	//	multiply_polynomials_sb_8(t1, t2, C1);
	//	multiply_polynomials_sb_8(A1, B1, C2);
	//}
	//else//if(n_2==4)
	//{
	//	multiply_polynomials_sb_4(A0, B0, C0);
	//	multiply_polynomials_sb_4(t1, t2, C1);
	//	multiply_polynomials_sb_4(A1, B1, C2);
	//}
	//else//n_2==2
	//{
	//	multiply_polynomials_sb_2(A0, B0, C0);
	//	multiply_polynomials_sb_2(t1, t2, C1);
	//	multiply_polynomials_sb_2(A1, B1, C2);
	//}
	//else//any n
	//{
	//	multiply_polynomials_sb(A0, B0, C0, n_2);
	//	multiply_polynomials_sb(t1, t2, C1, n_2);
	//	multiply_polynomials_sb(A1, B1, C2, n_2);
	//}

	for(int k=0;k<sub_size;++k)//interpolation
	{
		C1[k]-=C0[k]+C2[k];
		ab[k]+=C0[k];
		ab[k+n_2]+=C1[k];
		ab[k+n_2*2]+=C2[k];
	}
	memset(buffer, 0, buf_size*sizeof(short));
	delete[] buffer;
}
void		multiply_karatsuba(const short *a, const short *b, short *ab, int n, short logq, bool anti_cyclic)
{
	const int n_2=n>>1, sub_size=n-1, buf_size=sub_size*5;
	const short *A0=a, *A1=a+n_2, *B0=b, *B1=b+n_2;
	short *buffer=new short[buf_size],
		*C0=buffer, *C1=buffer+sub_size, *C2=buffer+sub_size*2,
		*t1=buffer+sub_size*3, *t2=buffer+sub_size*4;
	memset(buffer, 0, buf_size*sizeof(short));
	
	
#if PROCESSOR_ARCH>=AVX2
	if(n_2>=16)
		for(int k=0;k<n_2;k+=16)
		{
			__m256i a0=_mm256_loadu_si256((__m256i*)(A0+k));
			__m256i a1=_mm256_loadu_si256((__m256i*)(A1+k));
			__m256i b0=_mm256_loadu_si256((__m256i*)(B0+k));
			__m256i b1=_mm256_loadu_si256((__m256i*)(B1+k));
			__m256i v1=_mm256_add_epi16(a0, a1);
			__m256i v2=_mm256_add_epi16(b0, b1);
			_mm256_storeu_si256((__m256i*)(t1+k), v1);
			_mm256_storeu_si256((__m256i*)(t2+k), v2);
		}
	else
#elif PROCESSOR_ARCH>=SSE2
	if(n_2>=8)
		for(int k=0;k<n_2;k+=8)
		{
			__m128i a0=_mm_loadu_si128((__m128i*)(A0+k));
			__m128i a1=_mm_loadu_si128((__m128i*)(A1+k));
			__m128i b0=_mm_loadu_si128((__m128i*)(B0+k));
			__m128i b1=_mm_loadu_si128((__m128i*)(B1+k));
			__m128i v1=_mm_add_epi16(a0, a1);
			__m128i v2=_mm_add_epi16(b0, b1);
			_mm_storeu_si128((__m128i*)(t1+k), v1);
			_mm_storeu_si128((__m128i*)(t2+k), v2);
		}
	else
#endif//*/
		for(int k=0;k<n_2;++k)
		{
			t1[k]=A0[k]+A1[k];
			t2[k]=B0[k]+B1[k];
		}
	if(n_2>8)
	{
		multiply_karatsuba_noreduction(A0, B0, C0, n_2);//C0 = A(0)*B(0) = A0*B0
	//	multiply_polynomials_sb(A0, B0, C0, n_2, logq);

		multiply_karatsuba_noreduction(t1, t2, C1, n_2);//C1 = A(1)*B(1) = (A0+A1)(B0+B1)
	//	multiply_polynomials_sb(t1, t2, C1, n_2, logq);
	
		multiply_karatsuba_noreduction(A1, B1, C2, n_2);//C2 = A(inf)*B(inf) = A1*B1
	//	multiply_polynomials_sb(A1, B1, C2, n_2, logq);
	}
	else
	{
		multiply_polynomials_sb(A0, B0, C0, n_2);
		multiply_polynomials_sb(t1, t2, C1, n_2);
		multiply_polynomials_sb(A1, B1, C2, n_2);
	}
	
	int result_size=n*2-1;
	short *result=new short[result_size];
	memset(result, 0, result_size*sizeof(short));
	for(int k=0;k<sub_size;++k)//interpolation
	{
		C1[k]-=C0[k]+C2[k];
		result[k]+=C0[k];
		result[k+n_2]+=C1[k];
		result[k+n_2*2]+=C2[k];
	}

	short q_mask=(1<<logq)-1, sign_mask=-(short)anti_cyclic;
	for(int k=0;k<n-1;++k)//reduce mod x^n+1
		ab[k]=ab[k]+result[k]+(result[k+n]^sign_mask)-sign_mask & q_mask;
	ab[n-1]=ab[n-1]+result[n-1] & q_mask;

	memset(result, 0, result_size*sizeof(short));
	memset(buffer, 0, buf_size*sizeof(short));
	delete[] result, buffer;
}

#if PROCESSOR_ARCH>=AVX2
#define SCM_SIZE 16
__m256i temp;
__m256i c_avx[2*SCM_SIZE]; 
__m256i a[SCM_SIZE+2]; 
__m256i b[SCM_SIZE+2]; 
__m256i c_avx_extra[4];

__m256i mask,inv3_avx,inv9_avx,inv15_avx,int45_avx,int30_avx,int0_avx;

__m256i a_extra[2], b_extra[2];
void schoolbook_avx_new1()
{
	int i, j, scm_size_1=SCM_SIZE-1, vv;
	for(i=0; i<SCM_SIZE; i++)//the first triangle
	{
		c_avx[i]=_mm256_mullo_epi16 (a[0], b[i]);	
		for(j=1; j<=i; j++)
		{
			temp=_mm256_mullo_epi16 (a[j], b[i-j]);
			c_avx[i]=_mm256_add_epi16(c_avx[i], temp);
		}
	}
	for(i=1; i<SCM_SIZE; i++)//the second triangle
	{
		c_avx[scm_size_1+i] = _mm256_mullo_epi16 (a[i], b[scm_size_1]);	
		vv = scm_size_1+i;
		for(j=1; j<SCM_SIZE-i; j++)
		{
			temp = _mm256_mullo_epi16 (a[i+j], b[scm_size_1-j]);
			c_avx[vv] = _mm256_add_epi16(c_avx[vv], temp);
		}
	}
	c_avx[2*SCM_SIZE-1]=_mm256_setzero_si256();
}
void transpose(__m256i *M)//16x16 shorts
{
	int i;
	__m256i tL[8], tH[8];
	__m256i bL[4], bH[4], cL[4], cH[4];
	__m256i dL[2], dH[2], eL[2], eH[2], fL[2], fH[2], gL[2], gH[2];

	for(i=0; i<8; i=i+1)
	{
		tL[i] = _mm256_unpacklo_epi16(M[2*i], M[2*i+1]);//s: short, {as0, bs0, as1, bs1, as2, bs2, as3, bs3,	as8, bs8, as9, bs9, as10, bs10, as11, bs11}
		tH[i] = _mm256_unpackhi_epi16(M[2*i], M[2*i+1]);//			{as4, bs4, as5, bs5, as6, bs6, as7, bs7,	as12, bs12, as13, bs13, as14, bs14, as15, bs15}
	}
	for(i=0; i<4; i=i+1)
	{
		bL[i] = _mm256_unpacklo_epi32(tL[2*i], tL[2*i+1]); 
		bH[i] = _mm256_unpackhi_epi32(tL[2*i], tL[2*i+1]); 
	}
	for(i=0; i<4; i=i+1)
	{
		cL[i] = _mm256_unpacklo_epi32(tH[2*i], tH[2*i+1]); 
		cH[i] = _mm256_unpackhi_epi32(tH[2*i], tH[2*i+1]); 
	}
	for(i=0; i<2; i=i+1)
	{
		dL[i] = _mm256_unpacklo_epi64(bL[2*i], bL[2*i+1]); 
		dH[i] = _mm256_unpackhi_epi64(bL[2*i], bL[2*i+1]); 
	}
	for(i=0; i<2; i=i+1)
	{
		eL[i] = _mm256_unpacklo_epi64(bH[2*i], bH[2*i+1]); 
		eH[i] = _mm256_unpackhi_epi64(bH[2*i], bH[2*i+1]); 
	}

	for(i=0; i<2; i=i+1)
	{
		fL[i] = _mm256_unpacklo_epi64(cL[2*i], cL[2*i+1]); 
		fH[i] = _mm256_unpackhi_epi64(cL[2*i], cL[2*i+1]); 
	}
	for(i=0; i<2; i=i+1)
	{
		gL[i] = _mm256_unpacklo_epi64(cH[2*i], cH[2*i+1]); 
		gH[i] = _mm256_unpackhi_epi64(cH[2*i], cH[2*i+1]); 
	}
	M[0] = _mm256_permute2f128_si256(dL[0], dL[1], 0x20);//{a_lo, b_lo}
	M[8] = _mm256_permute2f128_si256(dL[0], dL[1], 0x31);//{a_hi, b_hi}
	M[1] = _mm256_permute2f128_si256(dH[0], dH[1], 0x20);//{a_lo, b_lo}
	M[9] = _mm256_permute2f128_si256(dH[0], dH[1], 0x31);//{a_hi, b_hi}

	M[2] = _mm256_permute2f128_si256(eL[0], eL[1], 0x20);//{a_lo, b_lo}
	M[10] = _mm256_permute2f128_si256(eL[0], eL[1], 0x31);//{a_hi, b_hi}
	M[3] = _mm256_permute2f128_si256(eH[0], eH[1], 0x20);//{a_lo, b_lo}
	M[11] = _mm256_permute2f128_si256(eH[0], eH[1], 0x31);//{a_hi, b_hi}

	M[4] = _mm256_permute2f128_si256(fL[0], fL[1], 0x20);//{a_lo, b_lo}
	M[12] = _mm256_permute2f128_si256(fL[0], fL[1], 0x31);//{a_hi, b_hi}
	M[5] = _mm256_permute2f128_si256(fH[0], fH[1], 0x20);//{a_lo, b_lo}
	M[13] = _mm256_permute2f128_si256(fH[0], fH[1], 0x31);

	M[6] = _mm256_permute2f128_si256(gL[0], gL[1], 0x20);//{a_lo, b_lo}
	M[14] = _mm256_permute2f128_si256(gL[0], gL[1], 0x31);//{a_hi, b_hi}
	M[7] = _mm256_permute2f128_si256(gH[0], gH[1], 0x20);//{a_lo, b_lo}
	M[15] = _mm256_permute2f128_si256(gH[0], gH[1], 0x31);//{a_hi, b_hi}
}
void karatsuba32_fork_avx_new(__m256i* a1, __m256i* b1, unsigned char position)
{
	a[position] = a1[0];
	b[position] = b1[0];

	if((position+1)>15)
	{ 
		a_extra[0] = a1[1];
		b_extra[0] = b1[1];	

		a_extra[1] = _mm256_add_epi16(a1[0], a1[1]);
		b_extra[1] = _mm256_add_epi16(b1[0], b1[1]);
	}
	else
	{
		a[position+1] = a1[1];
		b[position+1] = b1[1];

		a[position+2] = _mm256_add_epi16(a1[0], a1[1]);
		b[position+2] = _mm256_add_epi16(b1[0], b1[1]);
	}
}
void karatsuba32_fork_avx_partial(__m256i* a1, __m256i* b1, unsigned char position)
{
	a[position] = a1[1];
	b[position] = b1[1];

	a[position+1] = _mm256_add_epi16(a1[0], a1[1]);
	b[position+1] = _mm256_add_epi16(b1[0], b1[1]);
}
void karatsuba32_fork_avx_partial1(__m256i* a1, __m256i* b1, unsigned char position)
{
	a[position] = _mm256_add_epi16(a1[0], a1[1]);
	b[position] = _mm256_add_epi16(b1[0], b1[1]);
}
void karatsuba32_join_avx_new(__m256i* result_final, unsigned char position)
{
	result_final[0] = c_avx[position];   
	result_final[3] = c_avx[position+1+16];

	// b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]
	b[0] = _mm256_add_epi16(c_avx[position+16], c_avx[position+2]);

	// b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	b[1] = _mm256_add_epi16(c_avx[position+2+16], c_avx[position+1]);
		
	// b[0] = b[0] - a[0] - a[2]		
	b[2] = _mm256_sub_epi16(b[0], result_final[0]);
	result_final[1] = _mm256_sub_epi16(b[2], c_avx[position+1]);

	// b[1] = b[1] - a[1] - a[3]	
	b[2] = _mm256_sub_epi16(b[1], c_avx[position+16]);
	result_final[2] = _mm256_sub_epi16(b[2], result_final[3]);
}
void karatsuba32_join_avx_partial(__m256i* result_final, unsigned char position)
{
	// c_avx[position] --> c_avx_extra[0]
	// c_avx[position+16] --> c_avx_extra[1]
	// c_avx[position+1] --> c_avx[position]
	// c_avx[position+1+16] --> c_avx[position+16]
	// c_avx[position+2] --> c_avx[position+1]

	result_final[0] = c_avx_extra[0];   
	result_final[3] = c_avx[position+16];

	// b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]
	b[0] = _mm256_add_epi16(c_avx_extra[1], c_avx[position+1]);

	// b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	b[1] = _mm256_add_epi16(c_avx[position+1+16], c_avx[position]);
		
	// b[0] = b[0] - a[0] - a[2]		
	b[2] = _mm256_sub_epi16(b[0], result_final[0]);
	result_final[1] = _mm256_sub_epi16(b[2], c_avx[position]);

	// b[1] = b[1] - a[1] - a[3]		
	b[2] = _mm256_sub_epi16(b[1], c_avx_extra[1]);
	result_final[2] = _mm256_sub_epi16(b[2], result_final[3]);
}
void karatsuba32_join_avx_partial2(__m256i* result_final, unsigned char position)
{
	// c_avx[position] --> c_avx_extra[0]
	// c_avx[position+16] --> c_avx_extra[1]
	// c_avx[position+1] --> c_avx_extra[2]
	// c_avx[position+1+16] --> c_avx_extra[3]
	// c_avx[position+2] --> c_avx[position]
	// c_avx[position+2+16] --> c_avx[position+16]

	result_final[0] = c_avx_extra[0];   
	result_final[3] = c_avx_extra[3];

	// b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]
	b[0] = _mm256_add_epi16(c_avx_extra[1], c_avx[position]);

	// b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	b[1] = _mm256_add_epi16(c_avx[position+16], c_avx_extra[2]);
		
	// b[0] = b[0] - a[0] - a[2]		
	b[2] = _mm256_sub_epi16(b[0], result_final[0]);
	result_final[1] = _mm256_sub_epi16(b[2], c_avx_extra[2]);

	// b[1] = b[1] - a[1] - a[3]		
	b[2] = _mm256_sub_epi16(b[1], c_avx_extra[1]);
	result_final[2] = _mm256_sub_epi16(b[2], result_final[3]);
}
void join_32coefficient_results(__m256i result_d0[], __m256i result_d1[], __m256i result_d01[], __m256i result_64ks[])
{
	// {b[5],b[4]} = resultd0[63:32] + resultd01[31:0]
	b[4] = _mm256_add_epi16(result_d0[2], result_d01[0]);
	b[5] = _mm256_add_epi16(result_d0[3], result_d01[1]);

	// {b[7],b[6]} = resultd01[63:32] + resultd1[31:0]
	b[6] = _mm256_add_epi16(result_d01[2], result_d1[0]);
	b[7] = _mm256_add_epi16(result_d01[3], result_d1[1]);

	// {b[7],b[6],b[5],b[4]} <-- {b[7],b[6],b[5],b[4]} - {a[3],a[2],a[1],a[0]} - {a[7],a[6],a[5],a[4]}	
	result_64ks[2] = _mm256_sub_epi16(b[4], result_d0[0]);
	result_64ks[2] = _mm256_sub_epi16(result_64ks[2], result_d1[0]);
	result_64ks[3] = _mm256_sub_epi16(b[5], result_d0[1]);
	result_64ks[3] = _mm256_sub_epi16(result_64ks[3], result_d1[1]);
	result_64ks[4] = _mm256_sub_epi16(b[6], result_d0[2]);
	result_64ks[4] = _mm256_sub_epi16(result_64ks[4], result_d1[2]);
	result_64ks[5] = _mm256_sub_epi16(b[7], result_d0[3]);
	result_64ks[5] = _mm256_sub_epi16(result_64ks[5], result_d1[3]);

	result_64ks[0] = result_d0[0];
	result_64ks[1] = result_d0[1];
	result_64ks[6] = result_d1[2];
	result_64ks[7] = result_d1[3];
}
void batch_64coefficient_multiplications(
				 __m256i* a0, __m256i* b0, __m256i* result_final0,//4 registers x 16 shorts = 64 coefficients
				 __m256i* a1, __m256i* b1, __m256i* result_final1,
				 __m256i* a2, __m256i* b2, __m256i* result_final2,
				 __m256i* a3, __m256i* b3, __m256i* result_final3,
				 __m256i* a4, __m256i* b4, __m256i* result_final4,
				 __m256i* a5, __m256i* b5, __m256i* result_final5,
				 __m256i* a6, __m256i* b6, __m256i* result_final6)
{

	__m256i a_lu_temp[2], b_lu_temp[2];
	__m256i result_d0[16], result_d1[16], result_d01[16];
	unsigned short i;
	
	// KS splitting of 1st 64-coeff multiplication
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a0[i], a0[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b0[i], b0[2+i]);
	}
	karatsuba32_fork_avx_new(&a0[0], &b0[0], 0);	
	karatsuba32_fork_avx_new(&a0[2], &b0[2], 3);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 6);

	// KS splitting of 2nd 64-coeff multiplication
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a1[i], a1[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b1[i], b1[2+i]);
	}
	karatsuba32_fork_avx_new(&a1[0], &b1[0], 9);    
	karatsuba32_fork_avx_new(&a1[2], &b1[2], 12);  
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 15);	// Partial: loads only one of the three elements in the bucket

	// Compute 16 school-book multiplications in a batch.
	transpose(a);
	transpose(b);
	schoolbook_avx_new1();
	transpose(&c_avx[0]);
	transpose(&c_avx[16]);
		
	// store the partial multiplication result.
	c_avx_extra[0] = c_avx[15];
	c_avx_extra[1] = c_avx[15+16];

	karatsuba32_join_avx_new(result_d0, 0);
	karatsuba32_join_avx_new(result_d1, 3);
	karatsuba32_join_avx_new(result_d01, 6);
		
	// Final result of 1st 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final0);



	karatsuba32_join_avx_new(result_d0, 9);
	karatsuba32_join_avx_new(result_d1, 12);


	// Fork 2 parts of previous operands
	karatsuba32_fork_avx_partial(a_lu_temp, b_lu_temp, 0);    	 

	// Fork multiplication of a2*b2
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a2[i], a2[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b2[i], b2[2+i]);
	}
	karatsuba32_fork_avx_new(&a2[0], &b2[0], 2);   	
	karatsuba32_fork_avx_new(&a2[2], &b2[2], 5);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 8);

	// Fork multiplication of a3*b3
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a3[i], a3[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b3[i], b3[2+i]);
	}
	karatsuba32_fork_avx_new(&a3[0], &b3[0], 11);   	
	karatsuba32_fork_avx_new(&a3[2], &b3[2], 14);		// Partial: loads only two of the three elements in the bucket

	transpose(a);
	transpose(b);   	 
	schoolbook_avx_new1();

	transpose(&c_avx[0]);
	transpose(&c_avx[16]);

	karatsuba32_join_avx_partial(result_d01, 0);	// Combine results of this computation with previous computation
	// Final result of 2nd 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final1);

	// store the partial multiplication result. they will be combined after next batch multiplication
	c_avx_extra[0] = c_avx[14];
	c_avx_extra[1] = c_avx[14+16];
	c_avx_extra[2] = c_avx[15];
	c_avx_extra[3] = c_avx[15+16];




	karatsuba32_join_avx_new(result_d0, 2);
	karatsuba32_join_avx_new(result_d1, 5);
	karatsuba32_join_avx_new(result_d01, 8);
		
	// Final result of 3rd 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final2);

	// Join d0 of 4th 64-coeff multiplication
	karatsuba32_join_avx_new(result_d0, 11);



	// Fork 1 part of previous operands
	karatsuba32_fork_avx_partial1(&a3[2], &b3[2], 0);    	 
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 1);

	// Fork multiplication of a4*b4
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a4[i], a4[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b4[i], b4[2+i]);
	}
	karatsuba32_fork_avx_new(&a4[0], &b4[0], 4);   	
	karatsuba32_fork_avx_new(&a4[2], &b4[2], 7);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 10);

	// Fork multiplication of a5*b5
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a5[i], a5[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b5[i], b5[2+i]);
	}
	karatsuba32_fork_avx_new(&a5[0], &b5[0], 13);   	

	transpose(a);
	transpose(b);   	 
	schoolbook_avx_new1();

	transpose(&c_avx[0]);
	transpose(&c_avx[16]);

	karatsuba32_join_avx_partial2(result_d1, 0);
	karatsuba32_join_avx_new(result_d01, 1);

	// Final result of 4th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final3);




	karatsuba32_join_avx_new(result_d0, 4);
	karatsuba32_join_avx_new(result_d1, 7);
	karatsuba32_join_avx_new(result_d01, 10);

	// Final result of 5th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final4);

	karatsuba32_join_avx_new(result_d0, 13);
	


	// Fork remaining 2 parts of a5*b5
	karatsuba32_fork_avx_new(&a5[2], &b5[2], 0);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 3);
		
	// Fork multiplication of a6*b6
	for(i=0; i<2; i++)
	{
		a_lu_temp[i] = _mm256_add_epi16(a6[i], a6[2+i]);		
		b_lu_temp[i] = _mm256_add_epi16(b6[i], b6[2+i]);
	}

	karatsuba32_fork_avx_new(&a6[0], &b6[0], 6);   	
	karatsuba32_fork_avx_new(&a6[2], &b6[2], 9);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 12);

	transpose(a);
	transpose(b);   	 
	schoolbook_avx_new1();

	transpose(&c_avx[0]);
	transpose(&c_avx[16]);


	karatsuba32_join_avx_new(result_d1, 0);
	karatsuba32_join_avx_new(result_d01, 3);

	// Final result of 6th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final5);

	karatsuba32_join_avx_new(result_d0, 6);
	karatsuba32_join_avx_new(result_d1, 9);
	karatsuba32_join_avx_new(result_d01, 12);

	// Final result of 6th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final6);
}
#endif
/*#define SCM_SIZE 32
__m128i temp;
__m128i c_sse[2*SCM_SIZE]; 
__m128i a[SCM_SIZE+2]; 
__m128i b[SCM_SIZE+2]; 
__m128i c_sse_extra[8];
//__m128i c_sse_extra[4];

__m128i mask, inv3_sse, inv9_sse, inv15_sse, int45_sse, int30_sse, int0_sse;

__m128i a_extra[4], b_extra[4];
void		schoolbook_avx_new1()
{
	int i, j, scm_size_1=SCM_SIZE-1, vv;
	for(i=0; i<SCM_SIZE; i++)//the first triangle
	{
		c_sse[i]=_mm_mullo_epi16(a[0], b[i]);	
		for(j=1; j<=i; j++)
		{
			temp=_mm_mullo_epi16(a[j], b[i-j]);
			c_sse[i]=_mm_add_epi16(c_sse[i], temp);
		}
	}
	for(i=1; i<SCM_SIZE; i++)//the second triangle
	{
		c_sse[scm_size_1+i]=_mm_mullo_epi16(a[i], b[scm_size_1]);	
		vv = scm_size_1+i;
		for(j=1; j<SCM_SIZE-i; j++)
		{
			temp=_mm_mullo_epi16(a[i+j], b[scm_size_1-j]);
			c_sse[vv]=_mm_add_epi16(c_sse[vv], temp);
		}
	}
	c_sse[2*SCM_SIZE-1]=_mm_setzero_si128();
}
void		transpose8x8(__m128i *M)//8x8 shorts
{
	__m128i tL[4], tH[4];
	__m128i bL[2], bH[2], cL[2], cH[2];
	for(int i=0, j=0;i<4;++i, j+=2)
	{
		tL[i]=_mm_unpacklo_epi16(M[j], M[j+1]);//s: short,	{as0, bs0, as1, bs1, as2, bs2, as3, bs3}
		tH[i]=_mm_unpackhi_epi16(M[j], M[j+1]);//			{as4, bs4, as5, bs5, as6, bs6, as7, bs7}
	}
	for(int i=0, j=0;i<2;++i, j+=2)
	{
		bL[i]=_mm_unpacklo_epi32(tL[j], tL[j+1]); 
		bH[i]=_mm_unpackhi_epi32(tL[j], tL[j+1]); 
	}
	for(int i=0, j=0;i<2;++i, j+=2)
	{
		cL[i]=_mm_unpacklo_epi32(tH[j], tH[j+1]); 
		cH[i]=_mm_unpackhi_epi32(tH[j], tH[j+1]); 
	}
	M[0]=_mm_unpacklo_epi64(bL[0], bL[1]);
	M[1]=_mm_unpackhi_epi64(bL[0], bL[1]);
	M[2]=_mm_unpacklo_epi64(bH[0], bH[1]);
	M[3]=_mm_unpackhi_epi64(bH[0], bH[1]);
	M[4]=_mm_unpacklo_epi64(cL[0], cL[1]);
	M[5]=_mm_unpackhi_epi64(cL[0], cL[1]);
	M[6]=_mm_unpacklo_epi64(cH[0], cH[1]);
	M[7]=_mm_unpackhi_epi64(cH[0], cH[1]);
}
void		transpose(__m128i *M)//16x16 shorts
{
	transpose8x8(M), transpose8x8(M+8), transpose8x8(M+16), transpose8x8(M+24);
	__m128i t=M[8];	M[8]=M[16],	M[16]=t;
	t=M[8+1],	M[8+1]=M[16+1],	M[16+1]=t;
	t=M[8+2],	M[8+2]=M[16+2],	M[16+2]=t;
	t=M[8+3],	M[8+3]=M[16+3],	M[16+3]=t;
	t=M[8+4],	M[8+4]=M[16+4],	M[16+4]=t;
	t=M[8+5],	M[8+5]=M[16+5],	M[16+5]=t;
	t=M[8+6],	M[8+6]=M[16+6],	M[16+6]=t;
	t=M[8+7],	M[8+7]=M[16+7],	M[16+7]=t;
}
void		karatsuba32_fork_avx_new(__m128i *a1, __m128i *b1, unsigned char position)
{
	a[2*position]=a1[0], a[2*position+1]=a1[1];
	b[2*position]=b1[0], b[2*position+1]=b1[1];
	if((2*position+2)>30)
	{ 
		a_extra[0]=a1[2], a_extra[1]=a1[3];
		b_extra[0]=b1[2], b_extra[1]=b1[3];
		a_extra[2]=_mm_add_epi16(a1[0], a1[2]), a_extra[3]=_mm_add_epi16(a1[1], a1[3]);
		b_extra[2]=_mm_add_epi16(b1[0], b1[2]), b_extra[3]=_mm_add_epi16(b1[1], b1[3]);
	}
	else
	{
		a[2*position+2]=a1[2], a[2*position+3]=a1[3];
		b[2*position+2]=b1[2], b[2*position+3]=b1[3];
		a[2*position+4]=_mm_add_epi16(a1[0], a1[2]), a[2*position+5]=_mm_add_epi16(a1[1], a1[3]);
		b[2*position+4]=_mm_add_epi16(b1[0], b1[2]), b[2*position+5]=_mm_add_epi16(b1[1], b1[3]);
	}

	//a[position]=a1[0];
	//b[position]=b1[0];
	//if((position+1)>15)
	//{ 
	//	a_extra[0]=a1[1];
	//	b_extra[0]=b1[1];
	//	a_extra[1]=_mm_add_epi16(a1[0], a1[1]);
	//	b_extra[1]=_mm_add_epi16(b1[0], b1[1]);
	//}
	//else
	//{
	//	a[position+1]=a1[1];
	//	b[position+1]=b1[1];
	//	a[position+2]=_mm_add_epi16(a1[0], a1[1]);
	//	b[position+2]=_mm_add_epi16(b1[0], b1[1]);
	//}
}
void		karatsuba32_fork_avx_partial(__m128i *a1, __m128i *b1, unsigned char position)
{
	a[2*position]=a1[2];
	b[2*position]=b1[2];
	a[2*position+2]=_mm_add_epi16(a1[0], a1[2]);
	b[2*position+2]=_mm_add_epi16(b1[0], b1[2]);

	//a[position]=a1[1];
	//b[position]=b1[1];
	//a[position+1]=_mm_add_epi16(a1[0], a1[1]);
	//b[position+1]=_mm_add_epi16(b1[0], b1[1]);
}
void		karatsuba32_fork_avx_partial1(__m128i* a1, __m128i* b1, unsigned char position)
{
	a[2*position]=_mm_add_epi16(a1[0], a1[2]), a[2*position+1]=_mm_add_epi16(a1[1], a1[3]);
	b[2*position]=_mm_add_epi16(b1[0], b1[2]), b[2*position+1]=_mm_add_epi16(b1[1], b1[3]);

	//a[position]=_mm_add_epi16(a1[0], a1[1]);
	//b[position]=_mm_add_epi16(b1[0], b1[1]);
}
void		karatsuba32_join_avx_new(__m128i* result_final, unsigned char position)
{
	result_final[0]=c_sse[2*position], result_final[1]=c_sse[2*position+1];   
	result_final[6]=c_sse[2*position+2+32], result_final[7]=c_sse[2*position+3+33];

	b[0]=_mm_add_epi16(c_sse[2*position+32], c_sse[2*position+4]), b[1]=_mm_add_epi16(c_sse[2*position+33], c_sse[2*position+5]);	//b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]

	b[2]=_mm_add_epi16(c_sse[2*position+4+32], c_sse[2*position+2]), b[3]=_mm_add_epi16(c_sse[2*position+4+33], c_sse[2*position+3]);//b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	
	b[4]=_mm_sub_epi16(b[0], result_final[0]), b[5]=_mm_sub_epi16(b[0], result_final[0]);					//b[0] = b[0] - a[0] - a[2]
	result_final[2]=_mm_sub_epi16(b[4], c_sse[2*position+2]), result_final[3]=_mm_sub_epi16(b[5], c_sse[2*position+3]);

	b[4]=_mm_sub_epi16(b[2], c_sse[2*position+32]), b[5]=_mm_sub_epi16(b[3], c_sse[2*position+33]);				//b[1] = b[1] - a[1] - a[3]
	result_final[4]=_mm_sub_epi16(b[4], result_final[6]), result_final[5]=_mm_sub_epi16(b[5], result_final[7]);

	//result_final[0]=c_sse[position];   
	//result_final[3]=c_sse[position+1+16];

	//b[0]=_mm_add_epi16(c_sse[position+16], c_sse[position+2]);	//b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]

	//b[1]=_mm_add_epi16(c_sse[position+2+16], c_sse[position+1]);//b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	//
	//b[2]=_mm_sub_epi16(b[0], result_final[0]);					//b[0] = b[0] - a[0] - a[2]
	//result_final[1]=_mm_sub_epi16(b[2], c_sse[position+1]);

	//b[2]=_mm_sub_epi16(b[1], c_sse[position+16]);				//b[1] = b[1] - a[1] - a[3]
	//result_final[2]=_mm_sub_epi16(b[2], result_final[3]);
}
void		karatsuba32_join_avx_partial(__m128i* result_final, unsigned char position)
{
	// c_avx[position] --> c_avx_extra[0]
	// c_avx[position+16] --> c_avx_extra[1]
	// c_avx[position+1] --> c_avx[position]
	// c_avx[position+1+16] --> c_avx[position+16]
	// c_avx[position+2] --> c_avx[position+1]
	result_final[0]=c_sse_extra[0], result_final[1]=c_sse_extra[1];   
	result_final[6]=c_sse[2*position+32], result_final[7]=c_sse[2*position+33];

	b[0]=_mm_add_epi16(c_sse_extra[2], c_sse[2*position+2]), b[1]=_mm_add_epi16(c_sse_extra[3], c_sse[2*position+3]);		//b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]

	b[2]=_mm_add_epi16(c_sse[2*position+2+32], c_sse[2*position]), b[3]=_mm_add_epi16(c_sse[2*position+2+33], c_sse[2*position+1]);	//b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
			
	b[4]=_mm_sub_epi16(b[0], result_final[0]), b[5]=_mm_sub_epi16(b[1], result_final[1]);					//b[0] = b[0] - a[0] - a[2]
	result_final[2]=_mm_sub_epi16(b[4], c_sse[2*position]), result_final[3]=_mm_sub_epi16(b[5], c_sse[2*position+1]);

	b[4]=_mm_sub_epi16(b[2], c_sse_extra[2]), b[5]=_mm_sub_epi16(b[3], c_sse_extra[3]);					//b[1] = b[1] - a[1] - a[3]
	result_final[4]=_mm_sub_epi16(b[4], result_final[4]), result_final[5]=_mm_sub_epi16(b[5], result_final[5]);

	//result_final[0]=c_sse_extra[0];   
	//result_final[3]=c_sse[position+16];

	//b[0]=_mm_add_epi16(c_sse_extra[1], c_sse[position+1]);		//b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]

	//b[1]=_mm_add_epi16(c_sse[position+1+16], c_sse[position]);	//b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	//		
	//b[2]=_mm_sub_epi16(b[0], result_final[0]);					//b[0] = b[0] - a[0] - a[2]
	//result_final[1]=_mm_sub_epi16(b[2], c_sse[position]);

	//b[2]=_mm_sub_epi16(b[1], c_sse_extra[1]);					//b[1] = b[1] - a[1] - a[3]
	//result_final[2]=_mm_sub_epi16(b[2], result_final[3]);
}
void		karatsuba32_join_avx_partial2(__m128i* result_final, unsigned char position)
{
	// c_avx[position] --> c_avx_extra[0]
	// c_avx[position+16] --> c_avx_extra[1]
	// c_avx[position+1] --> c_avx_extra[2]
	// c_avx[position+1+16] --> c_avx_extra[3]
	// c_avx[position+2] --> c_avx[position]
	// c_avx[position+2+16] --> c_avx[position+16]
	result_final[0]=c_sse_extra[0], result_final[1]=c_sse_extra[1];
	result_final[6]=c_sse_extra[6], result_final[7]=c_sse_extra[7];

	b[0]=_mm_add_epi16(c_sse_extra[2], c_sse[2*position]), b[1]=_mm_add_epi16(c_sse_extra[3], c_sse[2*position+1]);	//b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]

	b[2]=_mm_add_epi16(c_sse[2*position+32], c_sse_extra[4]), b[3]=_mm_add_epi16(c_sse[2*position+33], c_sse_extra[5]);	//b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	
	b[4]=_mm_sub_epi16(b[0], result_final[0]), b[5]=_mm_sub_epi16(b[1], result_final[1]);				//b[0] = b[0] - a[0] - a[2]		
	result_final[2]=_mm_sub_epi16(b[4], c_sse_extra[4]), result_final[3]=_mm_sub_epi16(b[5], c_sse_extra[5]);
			
	b[4]=_mm_sub_epi16(b[2], c_sse_extra[2]), b[5]=_mm_sub_epi16(b[3], c_sse_extra[3]);				//b[1] = b[1] - a[1] - a[3]
	result_final[4]=_mm_sub_epi16(b[4], result_final[6]), result_final[5]=_mm_sub_epi16(b[5], result_final[7]);

	//result_final[0]=c_sse_extra[0];
	//result_final[3]=c_sse_extra[3];

	//b[0]=_mm_add_epi16(c_sse_extra[1], c_sse[position]);	//b[0] = resultd0[n-1:n/2] + resultd01[n/2-1:0]

	//b[1]=_mm_add_epi16(c_sse[position+16], c_sse_extra[2]);	//b[1] = resultd01[n-1:n/2] + resultd1[n/2-1:0]
	//
	//b[2]=_mm_sub_epi16(b[0], result_final[0]);				//b[0] = b[0] - a[0] - a[2]		
	//result_final[1]=_mm_sub_epi16(b[2], c_sse_extra[2]);
	//
	//b[2]=_mm_sub_epi16(b[1], c_sse_extra[1]);				//b[1] = b[1] - a[1] - a[3]
	//result_final[2]=_mm_sub_epi16(b[2], result_final[3]);
}
void		join_32coefficient_results(__m128i *result_d0, __m128i *result_d1, __m128i *result_d01, __m128i *result_64ks)
{
	b[8]=_mm_add_epi16(result_d0[4], result_d01[0]), b[9]=_mm_add_epi16(result_d0[5], result_d01[1]);	//{b[5],b[4]} = resultd0[63:32] + resultd01[31:0]
	b[10]=_mm_add_epi16(result_d0[6], result_d01[2]), b[11]=_mm_add_epi16(result_d0[7], result_d01[3]);

	b[12]=_mm_add_epi16(result_d01[4], result_d1[0]), b[13]=_mm_add_epi16(result_d01[5], result_d1[1]);	//{b[7],b[6]} = resultd01[63:32] + resultd1[31:0]
	b[14]=_mm_add_epi16(result_d01[6], result_d1[2]), b[15]=_mm_add_epi16(result_d01[7], result_d1[3]);

	result_64ks[4]=_mm_sub_epi16(b[8], result_d0[0]), result_64ks[5]=_mm_sub_epi16(b[9], result_d0[1]);	//{b[7],b[6],b[5],b[4]} <-- {b[7],b[6],b[5],b[4]} - {a[3],a[2],a[1],a[0]} - {a[7],a[6],a[5],a[4]}	
	result_64ks[4]=_mm_sub_epi16(result_64ks[4], result_d1[0]), result_64ks[5]=_mm_sub_epi16(result_64ks[5], result_d1[1]);
	result_64ks[6]=_mm_sub_epi16(b[10], result_d0[2]), result_64ks[7]=_mm_sub_epi16(b[11], result_d0[3]);
	result_64ks[6]=_mm_sub_epi16(result_64ks[6], result_d1[2]), result_64ks[7]=_mm_sub_epi16(result_64ks[7], result_d1[3]);
	result_64ks[8]=_mm_sub_epi16(b[12], result_d0[4]), result_64ks[9]=_mm_sub_epi16(b[13], result_d0[5]);
	result_64ks[8]=_mm_sub_epi16(result_64ks[8], result_d1[4]), result_64ks[9]=_mm_sub_epi16(result_64ks[9], result_d1[5]);
	result_64ks[10]=_mm_sub_epi16(b[14], result_d0[6]), result_64ks[11]=_mm_sub_epi16(b[15], result_d0[7]);
	result_64ks[10]=_mm_sub_epi16(result_64ks[10], result_d1[6]), result_64ks[11]=_mm_sub_epi16(result_64ks[11], result_d1[7]);

	result_64ks[0]=result_d0[0], result_64ks[1]=result_d0[1];
	result_64ks[2]=result_d0[2], result_64ks[3]=result_d0[3];
	result_64ks[12]=result_d1[4], result_64ks[13]=result_d1[5];
	result_64ks[14]=result_d1[6], result_64ks[15]=result_d1[7];

	//b[4]=_mm_add_epi16(result_d0[2], result_d01[0]);	//{b[5],b[4]} = resultd0[63:32] + resultd01[31:0]
	//b[5]=_mm_add_epi16(result_d0[3], result_d01[1]);

	//b[6]=_mm_add_epi16(result_d01[2], result_d1[0]);	//{b[7],b[6]} = resultd01[63:32] + resultd1[31:0]
	//b[7]=_mm_add_epi16(result_d01[3], result_d1[1]);

	//result_64ks[2]=_mm_sub_epi16(b[4], result_d0[0]);	//{b[7],b[6],b[5],b[4]} <-- {b[7],b[6],b[5],b[4]} - {a[3],a[2],a[1],a[0]} - {a[7],a[6],a[5],a[4]}	
	//result_64ks[2]=_mm_sub_epi16(result_64ks[2], result_d1[0]);
	//result_64ks[3]=_mm_sub_epi16(b[5], result_d0[1]);
	//result_64ks[3]=_mm_sub_epi16(result_64ks[3], result_d1[1]);
	//result_64ks[4]=_mm_sub_epi16(b[6], result_d0[2]);
	//result_64ks[4]=_mm_sub_epi16(result_64ks[4], result_d1[2]);
	//result_64ks[5]=_mm_sub_epi16(b[7], result_d0[3]);
	//result_64ks[5]=_mm_sub_epi16(result_64ks[5], result_d1[3]);

	//result_64ks[0]=result_d0[0];
	//result_64ks[1]=result_d0[1];
	//result_64ks[6]=result_d1[2];
	//result_64ks[7]=result_d1[3];
}
void		batch_64coefficient_multiplications(
			__m128i a0[8], __m128i b0[8], __m128i result_final0[16],//8 registers x 8 shorts = 64 coefficients
			__m128i a1[8], __m128i b1[8], __m128i result_final1[16],
			__m128i a2[8], __m128i b2[8], __m128i result_final2[16],
			__m128i a3[8], __m128i b3[8], __m128i result_final3[16],
			__m128i a4[8], __m128i b4[8], __m128i result_final4[16],
			__m128i a5[8], __m128i b5[8], __m128i result_final5[16],
			__m128i a6[8], __m128i b6[8], __m128i result_final6[16])
{
	__m128i a_lu_temp[4], b_lu_temp[4];
	__m128i result_d0[32], result_d1[32], result_d01[32];
	unsigned short i;
	
	//KS splitting of 1st 64-coeff multiplication
	for(i=0;i<4;++i)
	{
		a_lu_temp[i]=_mm_add_epi16(a0[i], a0[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b0[i], b0[4+i]);
	}
	karatsuba32_fork_avx_new(&a0[0], &b0[0], 0);
	karatsuba32_fork_avx_new(&a0[4], &b0[4], 3);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 6);
	
	//KS splitting of 2nd 64-coeff multiplication
	for(i=0;i<4;++i)
	{
		a_lu_temp[i]=_mm_add_epi16(a1[i], a1[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b1[i], b1[4+i]);
	}
	karatsuba32_fork_avx_new(&a1[0], &b1[0], 9);   
	karatsuba32_fork_avx_new(&a1[4], &b1[4], 12);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 15);//Partial: loads only one of the three elements in the bucket
	
	//Compute 16 school-book multiplications in a batch.
	transpose(a);
	transpose(b);
	schoolbook_avx_new1();
	transpose(&c_sse[0]);
	transpose(&c_sse[32]);
	
	c_sse_extra[0]=c_sse[30], c_sse_extra[1]=c_sse[31];//store the partial multiplication result.
	c_sse_extra[2]=c_sse[30+32], c_sse_extra[3]=c_sse[31+32];

	karatsuba32_join_avx_new(result_d0, 0);
	karatsuba32_join_avx_new(result_d1, 3);
	karatsuba32_join_avx_new(result_d01, 6);
		
	// Final result of 1st 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final0);

	karatsuba32_join_avx_new(result_d0, 9);
	karatsuba32_join_avx_new(result_d1, 12);

	// Fork 2 parts of previous operands
	karatsuba32_fork_avx_partial(a_lu_temp, b_lu_temp, 0);

	// Fork multiplication of a2*b2
	for(i=0; i<4; i++)
	{
		a_lu_temp[i]=_mm_add_epi16(a2[i], a2[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b2[i], b2[4+i]);
	}
	karatsuba32_fork_avx_new(&a2[0], &b2[0], 2);   	
	karatsuba32_fork_avx_new(&a2[4], &b2[4], 5);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 8);

	// Fork multiplication of a3*b3
	for(i=0; i<4; i++)
	{
		a_lu_temp[i]=_mm_add_epi16(a3[i], a3[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b3[i], b3[4+i]);
	}
	karatsuba32_fork_avx_new(&a3[0], &b3[0], 11);
	karatsuba32_fork_avx_new(&a3[4], &b3[4], 14);		// Partial: loads only two of the three elements in the bucket

	transpose(a);
	transpose(b);
	schoolbook_avx_new1();
	transpose(&c_sse[0]);
	transpose(&c_sse[32]);

	karatsuba32_join_avx_partial(result_d01, 0);	// Combine results of this computation with previous computation
	// Final result of 2nd 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final1);

	// store the partial multiplication result. they will be combined after next batch multiplication
	c_sse_extra[0]=c_sse[28], c_sse_extra[1]=c_sse[29];
	c_sse_extra[2]=c_sse[28+32], c_sse_extra[3]=c_sse[29+32];
	c_sse_extra[4]=c_sse[30], c_sse_extra[5]=c_sse[31];
	c_sse_extra[6]=c_sse[30+32], c_sse_extra[7]=c_sse[31+32];

	
	karatsuba32_join_avx_new(result_d0, 2);
	karatsuba32_join_avx_new(result_d1, 5);
	karatsuba32_join_avx_new(result_d01, 8);
		
	// Final result of 3rd 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final2);

	// Join d0 of 4th 64-coeff multiplication
	karatsuba32_join_avx_new(result_d0, 11);

	// Fork 1 part of previous operands
	karatsuba32_fork_avx_partial1(&a3[4], &b3[4], 0);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 1);

	// Fork multiplication of a4*b4
	for(i=0;i<4;++i)
	{
		a_lu_temp[i]=_mm_add_epi16(a4[i], a4[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b4[i], b4[4+i]);
	}
	karatsuba32_fork_avx_new(&a4[0], &b4[0], 4);   	
	karatsuba32_fork_avx_new(&a4[4], &b4[4], 7);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 10);

	// Fork multiplication of a5*b5
	for(i=0;i<4;++i)
	{
		a_lu_temp[i]=_mm_add_epi16(a5[i], a5[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b5[i], b5[4+i]);
	}
	karatsuba32_fork_avx_new(&a5[0], &b5[0], 13);

	transpose(a);
	transpose(b);
	schoolbook_avx_new1();

	transpose(&c_sse[0]);
	transpose(&c_sse[32]);

	karatsuba32_join_avx_partial2(result_d1, 0);
	karatsuba32_join_avx_new(result_d01, 1);

	// Final result of 4th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final3);
	

	karatsuba32_join_avx_new(result_d0, 4);
	karatsuba32_join_avx_new(result_d1, 7);
	karatsuba32_join_avx_new(result_d01, 10);

	// Final result of 5th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final4);

	karatsuba32_join_avx_new(result_d0, 13);
	
	// Fork remaining 2 parts of a5*b5
	karatsuba32_fork_avx_new(&a5[4], &b5[4], 0);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 3);
		
	// Fork multiplication of a6*b6
	for(i=0;i<4;++i)
	{
		a_lu_temp[i]=_mm_add_epi16(a6[i], a6[4+i]);
		b_lu_temp[i]=_mm_add_epi16(b6[i], b6[4+i]);
	}

	karatsuba32_fork_avx_new(&a6[0], &b6[0], 6);
	karatsuba32_fork_avx_new(&a6[4], &b6[4], 9);
	karatsuba32_fork_avx_new(a_lu_temp, b_lu_temp, 12);

	transpose(a);
	transpose(b);
	schoolbook_avx_new1();

	transpose(&c_sse[0]);
	transpose(&c_sse[32]);
	

	karatsuba32_join_avx_new(result_d1, 0);
	karatsuba32_join_avx_new(result_d01, 3);

	// Final result of 6th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final5);

	karatsuba32_join_avx_new(result_d0, 6);
	karatsuba32_join_avx_new(result_d1, 9);
	karatsuba32_join_avx_new(result_d01, 12);

	// Final result of 6th 64-coeff multiplication
	join_32coefficient_results(result_d0, result_d1, result_d01, result_final6);
}//*/
void		transpose8x8(__m128i *M)//8x8 shorts
{
	__m128i tL[4], tH[4];
	__m128i bL[2], bH[2], cL[2], cH[2];
	for(int i=0, j=0;i<4;++i, j+=2)
	{
		tL[i]=_mm_unpacklo_epi16(M[j], M[j+1]);//s: short,	{as0, bs0, as1, bs1, as2, bs2, as3, bs3}
		tH[i]=_mm_unpackhi_epi16(M[j], M[j+1]);//			{as4, bs4, as5, bs5, as6, bs6, as7, bs7}
	}
	for(int i=0, j=0;i<2;++i, j+=2)
	{
		bL[i]=_mm_unpacklo_epi32(tL[j], tL[j+1]); 
		bH[i]=_mm_unpackhi_epi32(tL[j], tL[j+1]); 
	}
	for(int i=0, j=0;i<2;++i, j+=2)
	{
		cL[i]=_mm_unpacklo_epi32(tH[j], tH[j+1]); 
		cH[i]=_mm_unpackhi_epi32(tH[j], tH[j+1]); 
	}
	M[0]=_mm_unpacklo_epi64(bL[0], bL[1]);
	M[1]=_mm_unpackhi_epi64(bL[0], bL[1]);
	M[2]=_mm_unpacklo_epi64(bH[0], bH[1]);
	M[3]=_mm_unpackhi_epi64(bH[0], bH[1]);
	M[4]=_mm_unpacklo_epi64(cL[0], cL[1]);
	M[5]=_mm_unpackhi_epi64(cL[0], cL[1]);
	M[6]=_mm_unpacklo_epi64(cH[0], cH[1]);
	M[7]=_mm_unpackhi_epi64(cH[0], cH[1]);
}
void		batch_32coefficient_multiplications(__m128i a[32], __m128i b[32], __m128i result[64])
{
	int n=32;
	for(int k=0;k<n;++k)
		for(int k2=0;k2<n;++k2)
		{
			__m128i t=_mm_mullo_epi16(a[k], b[k2]);
			result[k+k2]=_mm_add_epi16(result[k+k2], t);
		}
	//std::cout<<"result[0]:\t", print_register(result[0], 1<<13);//
}
void		batch_64coefficient_multiplications(__m128i a[64], __m128i b[64], __m128i result[128])
{
	const int n=64, n_2=n>>1, sub_size=n, buf_size=sub_size*3+n_2*2;
	__m128i *A0=a, *A1=a+n_2, *B0=b, *B1=b+n_2;
	__m128i *buffer=(__m128i*)_aligned_malloc(buf_size*sizeof(__m128i), sizeof(__m128i)),
		*C0=buffer, *C1=buffer+sub_size, *C2=buffer+sub_size*2,
		*t1=buffer+sub_size*3, *t2=buffer+sub_size*3+n_2;
	memset(buffer, 0, buf_size*sizeof(__m128i));
	for(int k=0;k<n_2;++k)
	{
		t1[k]=_mm_add_epi16(A0[k], A1[k]);
		t2[k]=_mm_add_epi16(B0[k], B1[k]);
	}
	batch_32coefficient_multiplications(A0, B0, C0);
	batch_32coefficient_multiplications(t1, t2, C1);
	batch_32coefficient_multiplications(A1, B1, C2);
//#ifdef _DEBUG
//	std::cout<<"A0:", print_element((short*)A0, n_2*8, 1<<13);//
//	std::cout<<"B0:", print_element((short*)B0, n_2*8, 1<<13);//
//	std::cout<<"C0:", print_element((short*)C0, n*8, 1<<13), std::cout<<endl;//
//	std::cout<<"t1:", print_element((short*)t1, n_2*8, 1<<13);//
//	std::cout<<"t2:", print_element((short*)t2, n_2*8, 1<<13);//
//	std::cout<<"C1:", print_element((short*)C1, n*8, 1<<13), std::cout<<endl;//
//	std::cout<<"A1:", print_element((short*)A1, n_2*8, 1<<13);//
//	std::cout<<"B1:", print_element((short*)B1, n_2*8, 1<<13);//
//	std::cout<<"C2:", print_element((short*)C2, n*8, 1<<13), std::cout<<endl;//
//#endif
	for(int k=0;k<sub_size;++k)
	{
		C1[k]=_mm_sub_epi16(C1[k], C0[k]);
		C1[k]=_mm_sub_epi16(C1[k], C2[k]);
		result[k]=_mm_add_epi16(result[k], C0[k]);
		result[k+n_2]=_mm_add_epi16(result[k+n_2], C1[k]);
		result[k+n_2*2]=_mm_add_epi16(result[k+n_2*2], C2[k]);
	}
	memset(buffer, 0, buf_size*sizeof(short));
	_aligned_free(buffer);//*/

	//const int n=64;
	//for(int k=0;k<n;++k)
	//	for(int k2=0;k2<n;++k2)
	//	{
	//		__m128i t=_mm_mullo_epi16(a[k], b[k2]);
	//		result[k+k2]=_mm_add_epi16(result[k+k2], t);
	//	}
}
void		multiply_toom_cook4_saber(const short *a, const short *b, short *ab, int n, short logq, bool anti_cyclic)
{
	const short inv3=-21845, inv9=-29127, inv15=-4369, int45=45, int30=30, int0=0;
//	const short inv3=43691, inv9=36409, inv15=61167, int45=45, int30=30, int0=0;
	const int n_4=n>>2, n_2=n>>1, n3_4=n_2+n_4;
	const short *A0=a, *A1=a+n_4, *A2=a+n_2, *A3=a+n3_4, *B0=b, *B1=b+n_4, *B2=b+n_2, *B3=b+n3_4;

	const int buf_size=n_2*17;
	short *buffer=new short[buf_size],
		*w1=buffer, *w2=buffer+n_2, *w3=buffer+n_2*2, *w4=buffer+n_2*3, *w5=buffer+n_2*4, *w6=buffer+n_2*5, *w7=buffer+n_2*6,
		*t1a=buffer+n_2*7, *t1b=buffer+n_2*8, *t2a=buffer+n_2*9, *t2b=buffer+n_2*10, *t3a=buffer+n_2*11, *t3b=buffer+n_2*12,
		*t4a=buffer+n_2*13, *t4b=buffer+n_2*14, *t5a=buffer+n_2*15, *t5b=buffer+n_2*16;
	memset(buffer, 0, buf_size*sizeof(short));
//#ifdef PROFILER
//	std::cout<<"TC4:\n";
//	long long t1=__rdtsc();
//#endif
#if PROCESSOR_ARCH>=AVX2
	if(n_4>=16)
	{
		const __m128i sh1=_mm_set_epi32(0, 0, 0, 1), sh2=_mm_set_epi32(0, 0, 0, 2), sh3=_mm_set_epi32(0, 0, 0, 3);
		for(int k=0;k<n_4;k+=16)
		{
			__m256i v0=_mm256_loadu_si256((__m256i*)(A0+k)), v1=_mm256_loadu_si256((__m256i*)(A1+k)), v2=_mm256_loadu_si256((__m256i*)(A2+k)), v3=_mm256_loadu_si256((__m256i*)(A3+k));
			__m256i v1s=_mm256_sll_epi16(v1, sh1), v2s=_mm256_sll_epi16(v2, sh2), v3s=_mm256_sll_epi16(v3, sh3);
			v1s=_mm256_add_epi16(v0, v1s);
			v1s=_mm256_add_epi16(v1s, v2s);
			v1s=_mm256_add_epi16(v1s, v3s);
			_mm256_storeu_si256((__m256i*)(t1a+k), v1s);
			__m256i sum1=_mm256_add_epi16(v0, v2);
			__m256i sum2=_mm256_add_epi16(v1, v3);
			v1s=_mm256_add_epi16(sum1, sum2);
			v2s=_mm256_sub_epi16(sum1, sum2);
			_mm256_storeu_si256((__m256i*)(t2a+k), v1s);
			_mm256_storeu_si256((__m256i*)(t3a+k), v2s);
			__m256i v0s=_mm256_sll_epi16(v0, sh3);
			v1s=_mm256_sll_epi16(v1, sh2);
			v2s=_mm256_sll_epi16(v2, sh1);
			sum1=_mm256_add_epi16(v0s, v2s);
			sum2=_mm256_add_epi16(v1s, v3);
			v1s=_mm256_add_epi16(sum1, sum2);
			v2s=_mm256_sub_epi16(sum1, sum2);
			_mm256_storeu_si256((__m256i*)(t4a+k), v1s);
			_mm256_storeu_si256((__m256i*)(t5a+k), v2s);

			v0=_mm256_loadu_si256((__m256i*)(B0+k)), v1=_mm256_loadu_si256((__m256i*)(B1+k)), v2=_mm256_loadu_si256((__m256i*)(B2+k)), v3=_mm256_loadu_si256((__m256i*)(B3+k));
			v1s=_mm256_sll_epi16(v1, sh1), v2s=_mm256_sll_epi16(v2, sh2), v3s=_mm256_sll_epi16(v3, sh3);
			v1s=_mm256_add_epi16(v0, v1s);
			v1s=_mm256_add_epi16(v1s, v2s);
			v1s=_mm256_add_epi16(v1s, v3s);
			_mm256_storeu_si256((__m256i*)(t1b+k), v1s);
			sum1=_mm256_add_epi16(v0, v2);
			sum2=_mm256_add_epi16(v1, v3);
			v1s=_mm256_add_epi16(sum1, sum2);
			v2s=_mm256_sub_epi16(sum1, sum2);
			_mm256_storeu_si256((__m256i*)(t2b+k), v1s);
			_mm256_storeu_si256((__m256i*)(t3b+k), v2s);
			v0s=_mm256_sll_epi16(v0, sh3);
			v1s=_mm256_sll_epi16(v1, sh2);
			v2s=_mm256_sll_epi16(v2, sh1);
			sum1=_mm256_add_epi16(v0s, v2s);
			sum2=_mm256_add_epi16(v1s, v3);
			v1s=_mm256_add_epi16(sum1, sum2);
			v2s=_mm256_sub_epi16(sum1, sum2);
			_mm256_storeu_si256((__m256i*)(t4b+k), v1s);
			_mm256_storeu_si256((__m256i*)(t5b+k), v2s);
		}
	}
	else
#elif PROCESSOR_ARCH>=SSE2
	if(n_4>=8)
	{
		const __m128i sh1=_mm_set_epi32(0, 0, 0, 1), sh2=_mm_set_epi32(0, 0, 0, 2), sh3=_mm_set_epi32(0, 0, 0, 3);
		for(int k=0;k<n_4;k+=8)
		{
			__m128i v0=_mm_loadu_si128((__m128i*)(A0+k)), v1=_mm_loadu_si128((__m128i*)(A1+k)), v2=_mm_loadu_si128((__m128i*)(A2+k)), v3=_mm_loadu_si128((__m128i*)(A3+k));
			__m128i v1s=_mm_sll_epi16(v1, sh1), v2s=_mm_sll_epi16(v2, sh2), v3s=_mm_sll_epi16(v3, sh3);
			v1s=_mm_add_epi16(v0, v1s);
			v1s=_mm_add_epi16(v1s, v2s);
			v1s=_mm_add_epi16(v1s, v3s);
			_mm_storeu_si128((__m128i*)(t1a+k), v1s);
			__m128i sum1=_mm_add_epi16(v0, v2);
			__m128i sum2=_mm_add_epi16(v1, v3);
			v1s=_mm_add_epi16(sum1, sum2);
			v2s=_mm_sub_epi16(sum1, sum2);
			_mm_storeu_si128((__m128i*)(t2a+k), v1s);
			_mm_storeu_si128((__m128i*)(t3a+k), v2s);
			__m128i v0s=_mm_sll_epi16(v0, sh3);
			v1s=_mm_sll_epi16(v1, sh2);
			v2s=_mm_sll_epi16(v2, sh1);
			sum1=_mm_add_epi16(v0s, v2s);
			sum2=_mm_add_epi16(v1s, v3);
			v1s=_mm_add_epi16(sum1, sum2);
			v2s=_mm_sub_epi16(sum1, sum2);
			_mm_storeu_si128((__m128i*)(t4a+k), v1s);
			_mm_storeu_si128((__m128i*)(t5a+k), v2s);

			v0=_mm_loadu_si128((__m128i*)(B0+k)), v1=_mm_loadu_si128((__m128i*)(B1+k)), v2=_mm_loadu_si128((__m128i*)(B2+k)), v3=_mm_loadu_si128((__m128i*)(B3+k));
			v1s=_mm_sll_epi16(v1, sh1), v2s=_mm_sll_epi16(v2, sh2), v3s=_mm_sll_epi16(v3, sh3);
			v1s=_mm_add_epi16(v0, v1s);
			v1s=_mm_add_epi16(v1s, v2s);
			v1s=_mm_add_epi16(v1s, v3s);
			_mm_storeu_si128((__m128i*)(t1b+k), v1s);
			sum1=_mm_add_epi16(v0, v2);
			sum2=_mm_add_epi16(v1, v3);
			v1s=_mm_add_epi16(sum1, sum2);
			v2s=_mm_sub_epi16(sum1, sum2);
			_mm_storeu_si128((__m128i*)(t2b+k), v1s);
			_mm_storeu_si128((__m128i*)(t3b+k), v2s);
			v0s=_mm_sll_epi16(v0, sh3);
			v1s=_mm_sll_epi16(v1, sh2);
			v2s=_mm_sll_epi16(v2, sh1);
			sum1=_mm_add_epi16(v0s, v2s);
			sum2=_mm_add_epi16(v1s, v3);
			v1s=_mm_add_epi16(sum1, sum2);
			v2s=_mm_sub_epi16(sum1, sum2);
			_mm_storeu_si128((__m128i*)(t4b+k), v1s);
			_mm_storeu_si128((__m128i*)(t5b+k), v2s);
		}
	}
	else
#endif
	for(int k=0;k<n_4;++k)
	{
		t1a[k]=A0[k]+(A1[k]<<1)+(A2[k]<<2)+(A3[k]<<3);
		short sum1=A0[k]+A2[k], sum2=A1[k]+A3[k];
		t2a[k]=sum1+sum2;
		t3a[k]=sum1-sum2;
		sum1=(A0[k]<<3)+(A2[k]<<1), sum2=(A1[k]<<2)+A3[k];
		t4a[k]=sum1+sum2;
		t5a[k]=sum1-sum2;

		t1b[k]=B0[k]+(B1[k]<<1)+(B2[k]<<2)+(B3[k]<<3);
		sum1=B0[k]+B2[k], sum2=B1[k]+B3[k];
		t2b[k]=sum1+sum2;
		t3b[k]=sum1-sum2;
		sum1=(B0[k]<<3)+(B2[k]<<1), sum2=(B1[k]<<2)+B3[k];
		t4b[k]=sum1+sum2;
		t5b[k]=sum1-sum2;

		//t1a[k]=A0[k]+(A1[k]<<1)+(A2[k]<<2)+(A3[k]<<3);
		//t2a[k]=A0[k]+A1[k]+A2[k]+A3[k];
		//t3a[k]=A0[k]-A1[k]+A2[k]-A3[k];
		//t4a[k]=(A0[k]<<3)+(A1[k]<<2)+(A2[k]<<1)+A3[k];
		//t5a[k]=(A0[k]<<3)-(A1[k]<<2)+(A2[k]<<1)-A3[k];
		//t1b[k]=B0[k]+(B1[k]<<1)+(B2[k]<<2)+(B3[k]<<3);
		//t2b[k]=B0[k]+B1[k]+B2[k]+B3[k];
		//t3b[k]=B0[k]-B1[k]+B2[k]-B3[k];
		//t4b[k]=(B0[k]<<3)+(B1[k]<<2)+(B2[k]<<1)+B3[k];
		//t5b[k]=(B0[k]<<3)-(B1[k]<<2)+(B2[k]<<1)-B3[k];
	}
//#ifdef PROFILER
//	std::cout<<"prepare temp:     "<<__rdtsc()-t1<<endl;
//	t1=__rdtsc();
//#endif
#if PROCESSOR_ARCH>=AVX2
	const int mb_size=7*(4+4+8);
	__m256i *m_buffer=(__m256i*)_aligned_malloc(mb_size*sizeof(__m256i), sizeof(__m256i)),
		*aa1=m_buffer     , *bb1=m_buffer+4   , *ww1=m_buffer+4*2,
		*aa2=m_buffer+4* 4, *bb2=m_buffer+4* 5, *ww2=m_buffer+4*6,
		*aa3=m_buffer+4* 8, *bb3=m_buffer+4* 9, *ww3=m_buffer+4*10,
		*aa4=m_buffer+4*12, *bb4=m_buffer+4*13, *ww4=m_buffer+4*14,
		*aa5=m_buffer+4*16, *bb5=m_buffer+4*17, *ww5=m_buffer+4*18,
		*aa6=m_buffer+4*20, *bb6=m_buffer+4*21, *ww6=m_buffer+4*22,
		*aa7=m_buffer+4*24, *bb7=m_buffer+4*25, *ww7=m_buffer+4*26;
	memset(m_buffer, 0, mb_size*sizeof(__m256i));
	//__m256i
	//	aa1[4], bb1[4], ww1[4],
	//	aa2[4], bb2[4], ww2[4],
	//	aa3[4], bb3[4], ww3[4],
	//	aa4[4], bb4[4], ww4[4],
	//	aa5[4], bb5[4], ww5[4],
	//	aa6[4], bb6[4], ww6[4],
	//	aa7[4], bb7[4], ww7[4];
	for(int k=0, k2=0;k<64;k+=16, ++k2)
	{
		aa1[k2]=_mm256_loadu_si256((__m256i*)(A3 +k)), bb1[k2]=_mm256_loadu_si256((__m256i*)(B3 +k));
		aa2[k2]=_mm256_loadu_si256((__m256i*)(t1a+k)), bb2[k2]=_mm256_loadu_si256((__m256i*)(t1b+k));
		aa3[k2]=_mm256_loadu_si256((__m256i*)(t2a+k)), bb3[k2]=_mm256_loadu_si256((__m256i*)(t2b+k));
		aa4[k2]=_mm256_loadu_si256((__m256i*)(t3a+k)), bb4[k2]=_mm256_loadu_si256((__m256i*)(t3b+k));
		aa5[k2]=_mm256_loadu_si256((__m256i*)(t4a+k)), bb5[k2]=_mm256_loadu_si256((__m256i*)(t4b+k));
		aa6[k2]=_mm256_loadu_si256((__m256i*)(t5a+k)), bb6[k2]=_mm256_loadu_si256((__m256i*)(t4b+k));
		aa7[k2]=_mm256_loadu_si256((__m256i*)(A0 +k)), bb7[k2]=_mm256_loadu_si256((__m256i*)(B0 +k));
	}
	batch_64coefficient_multiplications(aa1, bb1, ww1,  aa2, bb2, ww2,  aa3, bb3, ww3,  aa4, bb4, ww4,  aa5, bb5, ww5,  aa6, bb6, ww6,  aa7, bb7, ww7);
	for(int k=0, k2=0;k<128;k+=16, ++k2)
	{
		_mm256_storeu_si256((__m256i*)(w1+k), ww1[k2]);
		_mm256_storeu_si256((__m256i*)(w2+k), ww2[k2]);
		_mm256_storeu_si256((__m256i*)(w3+k), ww3[k2]);
		_mm256_storeu_si256((__m256i*)(w4+k), ww4[k2]);
		_mm256_storeu_si256((__m256i*)(w5+k), ww5[k2]);
		_mm256_storeu_si256((__m256i*)(w6+k), ww6[k2]);
		_mm256_storeu_si256((__m256i*)(w7+k), ww7[k2]);
	}
	_aligned_free(m_buffer);
#elif PROCESSOR_ARCH>=SSE2
	//const int mb_size=7*(8+8+16);
	//__m128i *m_buffer=(__m128i*)_aligned_malloc(mb_size*sizeof(__m128i), sizeof(__m128i)),
	//	*aa1=m_buffer     , *bb1=m_buffer+8   , *ww1=m_buffer+8* 2,
	//	*aa2=m_buffer+8* 4, *bb2=m_buffer+8* 5, *ww2=m_buffer+8* 6,
	//	*aa3=m_buffer+8* 8, *bb3=m_buffer+8* 9, *ww3=m_buffer+8*10,
	//	*aa4=m_buffer+8*12, *bb4=m_buffer+8*13, *ww4=m_buffer+8*14,
	//	*aa5=m_buffer+8*16, *bb5=m_buffer+8*17, *ww5=m_buffer+8*18,
	//	*aa6=m_buffer+8*20, *bb6=m_buffer+8*21, *ww6=m_buffer+8*22,
	//	*aa7=m_buffer+8*24, *bb7=m_buffer+8*25, *ww7=m_buffer+8*26;
	//memset(m_buffer, 0, mb_size*sizeof(__m128i));
	//for(int k=0, k2=0;k<64;k+=16, ++k2)
	//{
	//	aa1[k2]=_mm_loadu_si128((__m128i*)(A3 +k)), bb1[k2]=_mm_loadu_si128((__m128i*)(B3 +k));
	//	aa2[k2]=_mm_loadu_si128((__m128i*)(t1a+k)), bb2[k2]=_mm_loadu_si128((__m128i*)(t1b+k));
	//	aa3[k2]=_mm_loadu_si128((__m128i*)(t2a+k)), bb3[k2]=_mm_loadu_si128((__m128i*)(t2b+k));
	//	aa4[k2]=_mm_loadu_si128((__m128i*)(t3a+k)), bb4[k2]=_mm_loadu_si128((__m128i*)(t3b+k));
	//	aa5[k2]=_mm_loadu_si128((__m128i*)(t4a+k)), bb5[k2]=_mm_loadu_si128((__m128i*)(t4b+k));
	//	aa6[k2]=_mm_loadu_si128((__m128i*)(t5a+k)), bb6[k2]=_mm_loadu_si128((__m128i*)(t4b+k));
	//	aa7[k2]=_mm_loadu_si128((__m128i*)(A0 +k)), bb7[k2]=_mm_loadu_si128((__m128i*)(B0 +k));
	//}
	//batch_64coefficient_multiplications(aa1, bb1, ww1,  aa2, bb2, ww2,  aa3, bb3, ww3,  aa4, bb4, ww4,  aa5, bb5, ww5,  aa6, bb6, ww6,  aa7, bb7, ww7);
	//for(int k=0, k2=0;k<64;k+=16, ++k2)
	//{
	//	_mm_storeu_si128((__m128i*)(w1+k), ww1[k2]);
	//	_mm_storeu_si128((__m128i*)(w2+k), ww2[k2]);
	//	_mm_storeu_si128((__m128i*)(w3+k), ww3[k2]);
	//	_mm_storeu_si128((__m128i*)(w4+k), ww4[k2]);
	//	_mm_storeu_si128((__m128i*)(w5+k), ww5[k2]);
	//	_mm_storeu_si128((__m128i*)(w6+k), ww6[k2]);
	//	_mm_storeu_si128((__m128i*)(w7+k), ww7[k2]);
	//}
	//_aligned_free(m_buffer);
	
	const int mb_size=n_4+n_4+n_2;
	__m128i *m_buffer=(__m128i*)_aligned_malloc(mb_size*sizeof(__m128i), sizeof(__m128i)),
		*ma=m_buffer, *mb=m_buffer+n_4, *mr=m_buffer+n_4*2;
	memset(mr, 0, n_2*sizeof(__m128i));
	for(int k=0;k<n_4;++k)
	{
		ma[k]=_mm_set_epi16(0, A0[k], t5a[k], t4a[k], t3a[k], t2a[k], t1a[k], A3[k]);
		mb[k]=_mm_set_epi16(0, B0[k], t5b[k], t4b[k], t3b[k], t2b[k], t1b[k], B3[k]);
		//ma[k]=_mm_setr_epi16(A3[k], t1a[k], t2a[k], t3a[k], t4a[k], t5a[k], A0[k], 0);
		//mb[k]=_mm_setr_epi16(B3[k], t1b[k], t2b[k], t3b[k], t4b[k], t5b[k], B0[k], 0);
	}
	batch_64coefficient_multiplications(ma, mb, mr);
	for(int k=0;k<n_2;k+=8)
	{
		transpose8x8(mr+k);
		_mm_storeu_si128((__m128i*)(w1+k), mr[k]);
		_mm_storeu_si128((__m128i*)(w2+k), mr[k+1]);
		_mm_storeu_si128((__m128i*)(w3+k), mr[k+2]);
		_mm_storeu_si128((__m128i*)(w4+k), mr[k+3]);
		_mm_storeu_si128((__m128i*)(w5+k), mr[k+4]);
		_mm_storeu_si128((__m128i*)(w6+k), mr[k+5]);
		_mm_storeu_si128((__m128i*)(w7+k), mr[k+6]);
	}
	//for(int k=0;k<n_2;++k)
	//	w1[k]=mr[k].m128i_i16[0], w2[k]=mr[k].m128i_i16[1], w3[k]=mr[k].m128i_i16[2], w4[k]=mr[k].m128i_i16[3], w5[k]=mr[k].m128i_i16[4], w6[k]=mr[k].m128i_i16[5], w7[k]=mr[k].m128i_i16[6];
	_aligned_free(m_buffer);
#else//*/
	multiply_karatsuba_noreduction(A3, B3, w1, n_4);//w1 = A(inf)*B(inf) = A3*B3		7 muls: ~100K cycles
	multiply_karatsuba_noreduction(t1a, t1b, w2, n_4);//w2 = A(2)*B(2) = (A0+2A1+4A2+8A3)(B0+2B1+4B2+8B3)
	multiply_karatsuba_noreduction(t2a, t2b, w3, n_4);//w3 = A(1)*B(1) = (A0+A1+A2+A3)(B0+B1+B2+B3)
	multiply_karatsuba_noreduction(t3a, t3b, w4, n_4);//w4 = A(-1)*B(-1) = (A0-A1+A2-A3)(B0-B1+B2-B3)
	multiply_karatsuba_noreduction(t4a, t4b, w5, n_4);//w5 = A(1/2)*B(1/2) = (8*A0+4*A1+2*A2+A3)(8*B0+4*B1+2*B2+B3)
	multiply_karatsuba_noreduction(t5a, t5b, w6, n_4);//w6 = A(-1/2)*B(-1/2) = (8*A0-4*A1+2*A2-A3)(8*B0-4*B1+2*B2-B3)
	multiply_karatsuba_noreduction(A0, B0, w7, n_4);//w7 = A(0)*B(0) = A0*B0
#endif
//#ifdef PROFILER
//	std::cout<<"7 muls:           "<<__rdtsc()-t1<<endl;
//	t1=__rdtsc();
//#endif

/*	const int sub_size=n_4*2-1, buf_size=sub_size*9;
	short *buffer=new short[buf_size],
		*w1=buffer, *w2=buffer+sub_size, *w3=buffer+sub_size*2, *w4=buffer+sub_size*3, *w5=buffer+sub_size*4, *w6=buffer+sub_size*5, *w7=buffer+sub_size*6,
		*t1=buffer+sub_size*7, *t2=buffer+sub_size*8;
	memset(buffer, 0, buf_size*sizeof(short));

	multiply_karatsuba_noreduction(A3, B3, w1, n_4);//w1 = A(inf)*B(inf) = A3*B3
//	multiply_polynomials_sb(A3, B3, w1, n_4);

	for(int k=0;k<n_4;++k)//w2 = A(2)*B(2) = (A0+2A1+4A2+8A3)(B0+2B1+4B2+8B3)
	{
		t1[k]=A0[k]+(A1[k]<<1)+(A2[k]<<2)+(A3[k]<<3);
		t2[k]=B0[k]+(B1[k]<<1)+(B2[k]<<2)+(B3[k]<<3);
	}
	multiply_karatsuba_noreduction(t1, t2, w2, n_4);
//	multiply_polynomials_sb(t1, t2, w2, n_4);

	for(int k=0;k<n_4;++k)//w3 = A(1)*B(1) = (A0+A1+A2+A3)(B0+B1+B2+B3)
	{
		t1[k]=A0[k]+A1[k]+A2[k]+A3[k];
		t2[k]=B0[k]+B1[k]+B2[k]+B3[k];
	}
	multiply_karatsuba_noreduction(t1, t2, w3, n_4);
//	multiply_polynomials_sb(t1, t2, w3, n_4);

	for(int k=0;k<n_4;++k)//w4 = A(-1)*B(-1) = (A0-A1+A2-A3)(B0-B1+B2-B3)
	{
		t1[k]=A0[k]-A1[k]+A2[k]-A3[k];
		t2[k]=B0[k]-B1[k]+B2[k]-B3[k];
	}
	multiply_karatsuba_noreduction(t1, t2, w4, n_4);
//	multiply_polynomials_sb(t1, t2, w4, n_4);

	for(int k=0;k<n_4;++k)//w5 = A(1/2)*B(1/2) = (8*A0+4*A1+2*A2+A3)(8*B0+4*B1+2*B2+B3)
	{
		t1[k]=(A0[k]<<3)+(A1[k]<<2)+(A2[k]<<1)+A3[k];
		t2[k]=(B0[k]<<3)+(B1[k]<<2)+(B2[k]<<1)+B3[k];
	}
	multiply_karatsuba_noreduction(t1, t2, w5, n_4);
//	multiply_polynomials_sb(t1, t2, w5, n_4);

	for(int k=0;k<n_4;++k)//w6 = A(-1/2)*B(-1/2) = (8*A0-4*A1+2*A2-A3)(8*B0-4*B1+2*B2-B3)
	{
		t1[k]=(A0[k]<<3)-(A1[k]<<2)+(A2[k]<<1)-A3[k];
		t2[k]=(B0[k]<<3)-(B1[k]<<2)+(B2[k]<<1)-B3[k];
	}
	multiply_karatsuba_noreduction(t1, t2, w6, n_4);
//	multiply_polynomials_sb(t1, t2, w6, n_4);

	multiply_karatsuba_noreduction(A0, B0, w7, n_4);//w7 = A(0)*B(0) = A0*B0
//	multiply_polynomials_sb(A0, B0, w7, n_4);//*/
//#ifdef _DEBUG
//	std::cout<<"A3:", print_element(A3, n_4, 1<<logq);//
//	std::cout<<"B3:", print_element(B3, n_4, 1<<logq);//
//	std::cout<<"w1:", print_element(w1, n_2, 1<<logq), std::cout<<endl;//
//
//	std::cout<<"t1a:", print_element(t1a, n_4, 1<<logq);//
//	std::cout<<"t1b:", print_element(t1b, n_4, 1<<logq);//
//	std::cout<<"w2:", print_element(w2, n_2, 1<<logq), std::cout<<endl;//
//	std::cout<<"t2a:", print_element(t2a, n_4, 1<<logq);//
//	std::cout<<"t2b:", print_element(t2b, n_4, 1<<logq);//
//	std::cout<<"w3:", print_element(w3, n_2, 1<<logq), std::cout<<endl;//
//	std::cout<<"t3a:", print_element(t3a, n_4, 1<<logq);//
//	std::cout<<"t3b:", print_element(t3b, n_4, 1<<logq);//
//	std::cout<<"w4:", print_element(w4, n_2, 1<<logq), std::cout<<endl;//
//	std::cout<<"t4a:", print_element(t4a, n_4, 1<<logq);//
//	std::cout<<"t4b:", print_element(t4b, n_4, 1<<logq);//
//	std::cout<<"w5:", print_element(w5, n_2, 1<<logq), std::cout<<endl;//
//	std::cout<<"t5a:", print_element(t5a, n_4, 1<<logq);//
//	std::cout<<"t5b:", print_element(t5b, n_4, 1<<logq);//
//	std::cout<<"w6:", print_element(w6, n_2, 1<<logq), std::cout<<endl;//
//
//	std::cout<<"A0:", print_element(A0, n_4, 1<<logq);//
//	std::cout<<"B0:", print_element(B0, n_4, 1<<logq);//
//	std::cout<<"w7:", print_element(w7, n_2, 1<<logq);//
//#endif

	int result_size=n<<1;
//	int result_size=n*2-1;
	short *result=new short[result_size];
	memset(result, 0, result_size*sizeof(short));
#if PROCESSOR_ARCH>=AVX2
	if(n_2>=16)
	{
		const __m128i sh1=_mm_set_epi32(0, 0, 0, 1), sh2=_mm_set_epi32(0, 0, 0, 2), sh3=_mm_set_epi32(0, 0, 0, 3), sh4=_mm_set_epi32(0, 0, 0, 4), sh6=_mm_set_epi32(0, 0, 0, 6);
		const __m256i m45=_mm256_set1_epi16(45), m30=_mm256_set1_epi16(30), m_inv3=_mm256_set1_epi16(inv3), m_inv9=_mm256_set1_epi16(inv9), m_inv15=_mm256_set1_epi16(inv15);
		for(int k=0;k<n_2;k+=16)
		{
			__m256i w1k=_mm256_loadu_si256((__m256i*)(w1+k)), w2k=_mm256_loadu_si256((__m256i*)(w2+k)), w3k=_mm256_loadu_si256((__m256i*)(w3+k)), w4k=_mm256_loadu_si256((__m256i*)(w4+k)),
				w5k=_mm256_loadu_si256((__m256i*)(w5+k)), w6k=_mm256_loadu_si256((__m256i*)(w6+k)), w7k=_mm256_loadu_si256((__m256i*)(w7+k));
			w2k=_mm256_add_epi16(w2k, w5k);//w2 += w5

			w6k=_mm256_sub_epi16(w6k, w5k);//w6 -= w5

			w4k=_mm256_sub_epi16(w4k, w3k);//w4 = (w4-w3)/2
			w4k=_mm256_sra_epi16(w4k, sh1);

			__m256i temp=_mm256_sll_epi16(w7k, sh6);//w5 -= w1+64w7
			w5k=_mm256_sub_epi16(w5k, w1k);
			w5k=_mm256_sub_epi16(w5k, temp);

			w3k=_mm256_add_epi16(w3k, w4k);//w3 += w4

			w5k=_mm256_sll_epi16(w5k, sh1);//w5 = 2w5+w6
			w5k=_mm256_add_epi16(w5k, w6k);

			temp=_mm256_sll_epi16(w3k, sh6);//w2 -= 65w3
			temp=_mm256_add_epi16(temp, w3k);
			w2k=_mm256_sub_epi16(w2k, temp);

			w3k=_mm256_sub_epi16(w3k, w7k);//w3 -= w7+w1
			w3k=_mm256_sub_epi16(w3k, w1k);

			temp=_mm256_mullo_epi16(w3k, m45);//w2 += 45w3
			w2k=_mm256_add_epi16(w2k, temp);

			temp=_mm256_sll_epi16(w3k, sh3);//w5 = (w5-8w3)/24
			w5k=_mm256_sub_epi16(w5k, temp);
			w5k=_mm256_mullo_epi16(w5k, m_inv3);
			w5k=_mm256_sra_epi16(w5k, sh3);

			w6k=_mm256_add_epi16(w6k, w2k);//w6 += w2

			temp=_mm256_sll_epi16(w4k, sh4);//w2 = (w2+16w4)/18
			w2k=_mm256_add_epi16(w2k, temp);
			w2k=_mm256_mullo_epi16(w2k, m_inv9);
			w2k=_mm256_sra_epi16(w2k, sh1);

			w3k=_mm256_sub_epi16(w3k, w5k);//w3 -= w5

			w4k=_mm256_add_epi16(w4k, w2k);//w4 = -(w4+w3)
			w4k=_mm256_sub_epi16(_mm256_setzero_si256(), w4k);

			temp=_mm256_mullo_epi16(w2k, m30);//w6 = (30w2-w6)/60
			w6k=_mm256_sub_epi16(temp, w6k);
			w6k=_mm256_mullo_epi16(w6k, m_inv15);
			w6k=_mm256_sra_epi16(w6k, sh2);

			w2k=_mm256_sub_epi16(w2k, w6k);//w2 -= w6

			temp=_mm256_loadu_si256((__m256i*)(result+k			)), w7k=_mm256_add_epi16(w7k, temp), _mm256_storeu_si256((__m256i*)(result+k		), w7k);
			temp=_mm256_loadu_si256((__m256i*)(result+k+n_4		)), w6k=_mm256_add_epi16(w6k, temp), _mm256_storeu_si256((__m256i*)(result+k+n_4	), w6k);
			temp=_mm256_loadu_si256((__m256i*)(result+k+n_4*2	)), w5k=_mm256_add_epi16(w5k, temp), _mm256_storeu_si256((__m256i*)(result+k+n_4*2	), w5k);
			temp=_mm256_loadu_si256((__m256i*)(result+k+n_4*3	)), w4k=_mm256_add_epi16(w4k, temp), _mm256_storeu_si256((__m256i*)(result+k+n_4*3	), w4k);
			temp=_mm256_loadu_si256((__m256i*)(result+k+n_4*4	)), w3k=_mm256_add_epi16(w3k, temp), _mm256_storeu_si256((__m256i*)(result+k+n_4*4	), w3k);
			temp=_mm256_loadu_si256((__m256i*)(result+k+n_4*5	)), w2k=_mm256_add_epi16(w2k, temp), _mm256_storeu_si256((__m256i*)(result+k+n_4*5	), w2k);
			temp=_mm256_loadu_si256((__m256i*)(result+k+n_4*6	)), w1k=_mm256_add_epi16(w1k, temp), _mm256_storeu_si256((__m256i*)(result+k+n_4*6	), w1k);
		}
	}
	else
#elif PROCESSOR_ARCH>=SSE2
	if(n_2>=8)
	{
		const __m128i sh1=_mm_set_epi32(0, 0, 0, 1), sh2=_mm_set_epi32(0, 0, 0, 2), sh3=_mm_set_epi32(0, 0, 0, 3), sh4=_mm_set_epi32(0, 0, 0, 4), sh6=_mm_set_epi32(0, 0, 0, 6),
			m45=_mm_set1_epi16(45), m30=_mm_set1_epi16(30), m_inv3=_mm_set1_epi16(inv3), m_inv9=_mm_set1_epi16(inv9), m_inv15=_mm_set1_epi16(inv15);
		for(int k=0;k<n_2;k+=8)
		{
			__m128i w1k=_mm_loadu_si128((__m128i*)(w1+k)), w2k=_mm_loadu_si128((__m128i*)(w2+k)), w3k=_mm_loadu_si128((__m128i*)(w3+k)), w4k=_mm_loadu_si128((__m128i*)(w4+k)),
				w5k=_mm_loadu_si128((__m128i*)(w5+k)), w6k=_mm_loadu_si128((__m128i*)(w6+k)), w7k=_mm_loadu_si128((__m128i*)(w7+k));
			w2k=_mm_add_epi16(w2k, w5k);//w2 += w5

			w6k=_mm_sub_epi16(w6k, w5k);//w6 -= w5

			w4k=_mm_sub_epi16(w4k, w3k);//w4 = (w4-w3)/2
			w4k=_mm_sra_epi16(w4k, sh1);

			__m128i temp=_mm_sll_epi16(w7k, sh6);//w5 -= w1+64w7
			w5k=_mm_sub_epi16(w5k, w1k);
			w5k=_mm_sub_epi16(w5k, temp);

			w3k=_mm_add_epi16(w3k, w4k);//w3 += w4

			w5k=_mm_sll_epi16(w5k, sh1);//w5 = 2w5+w6
			w5k=_mm_add_epi16(w5k, w6k);

			temp=_mm_sll_epi16(w3k, sh6);//w2 -= 65w3
			temp=_mm_add_epi16(temp, w3k);
			w2k=_mm_sub_epi16(w2k, temp);

			w3k=_mm_sub_epi16(w3k, w7k);//w3 -= w7+w1
			w3k=_mm_sub_epi16(w3k, w1k);

			temp=_mm_mullo_epi16(w3k, m45);//w2 += 45w3
			w2k=_mm_add_epi16(w2k, temp);

			temp=_mm_sll_epi16(w3k, sh3);//w5 = (w5-8w3)/24
			w5k=_mm_sub_epi16(w5k, temp);
			w5k=_mm_mullo_epi16(w5k, m_inv3);
			w5k=_mm_sra_epi16(w5k, sh3);

			w6k=_mm_add_epi16(w6k, w2k);//w6 += w2

			temp=_mm_sll_epi16(w4k, sh4);//w2 = (w2+16w4)/18
			w2k=_mm_add_epi16(w2k, temp);
			w2k=_mm_mullo_epi16(w2k, m_inv9);
			w2k=_mm_sra_epi16(w2k, sh1);

			w3k=_mm_sub_epi16(w3k, w5k);//w3 -= w5

			w4k=_mm_add_epi16(w4k, w2k);//w4 = -(w4+w3)
			w4k=_mm_sub_epi16(_mm_setzero_si128(), w4k);

			temp=_mm_mullo_epi16(w2k, m30);//w6 = (30w2-w6)/60
			w6k=_mm_sub_epi16(temp, w6k);
			w6k=_mm_mullo_epi16(w6k, m_inv15);
			w6k=_mm_sra_epi16(w6k, sh2);

			w2k=_mm_sub_epi16(w2k, w6k);//w2 -= w6

			temp=_mm_loadu_si128((__m128i*)(result+k		)), w7k=_mm_add_epi16(w7k, temp), _mm_storeu_si128((__m128i*)(result+k		), w7k);
			temp=_mm_loadu_si128((__m128i*)(result+k+n_4	)), w6k=_mm_add_epi16(w6k, temp), _mm_storeu_si128((__m128i*)(result+k+n_4	), w6k);
			temp=_mm_loadu_si128((__m128i*)(result+k+n_4*2	)), w5k=_mm_add_epi16(w5k, temp), _mm_storeu_si128((__m128i*)(result+k+n_4*2), w5k);
			temp=_mm_loadu_si128((__m128i*)(result+k+n_4*3	)), w4k=_mm_add_epi16(w4k, temp), _mm_storeu_si128((__m128i*)(result+k+n_4*3), w4k);
			temp=_mm_loadu_si128((__m128i*)(result+k+n_4*4	)), w3k=_mm_add_epi16(w3k, temp), _mm_storeu_si128((__m128i*)(result+k+n_4*4), w3k);
			temp=_mm_loadu_si128((__m128i*)(result+k+n_4*5	)), w2k=_mm_add_epi16(w2k, temp), _mm_storeu_si128((__m128i*)(result+k+n_4*5), w2k);
			temp=_mm_loadu_si128((__m128i*)(result+k+n_4*6	)), w1k=_mm_add_epi16(w1k, temp), _mm_storeu_si128((__m128i*)(result+k+n_4*6), w1k);
		}
	}
	else
#endif//*/
	for(int k=0;k<n_2;++k)//interpolation
//	for(int k=0;k<sub_size;++k)
	{
		//w2[k]+=w5[k];
		//w6[k]-=w5[k];
		//w4[k]=(w4[k]-w3[k])>>1;
		//w5[k]-=w1[k]+(w7[k]<<6);
		//w3[k]+=w4[k];
		//w5[k]=(w5[k]<<1)+w6[k];
		//w2[k]-=(w3[k]<<6)+w3[k];
		//w3[k]-=w7[k]+w1[k];
		//w2[k]+=45*w3[k];
		//w5[k]=(w5[k]-(w3[k]<<3))*inv3>>3;
		//w6[k]+=w2[k];
		//w2[k]=(w2[k]+(w4[k]<<4))*inv9>>1;
		//w3[k]-=w5[k];
		//w4[k]=-(w4[k]+w2[k]);
		//w6[k]=(30*w2[k]-w6[k])*inv15>>2;
		//w2[k]-=w6[k];

		w2[k]=w2[k]+w5[k];
		w6[k]=w6[k]-w5[k];
		w4[k]=short(w4[k]-w3[k])>>1;
		w5[k]=w5[k]-w1[k]-(w7[k]<<6);
		w3[k]=w3[k]+w4[k];
		w5[k]=(w5[k]<<1)+w6[k];
		w2[k]=w2[k]-(w3[k]<<6)-w3[k];
		w3[k]=w3[k]-w7[k]-w1[k];
		w2[k]=w2[k]+45*w3[k];
		w5[k]=short((w5[k]-(w3[k]<<3))*inv3)>>3;
		w6[k]=w6[k]+w2[k];
		w2[k]=short((w2[k]+(w4[k]<<4))*inv9)>>1;
		w3[k]=w3[k]-w5[k];
		w4[k]=-(w4[k]+w2[k]);
		w6[k]=short((30*w2[k]-w6[k])*inv15)>>2;
		w2[k]=w2[k]-w6[k];

		result[k]+=w7[k];
		result[k+n_4]+=w6[k];
		result[k+n_4*2]+=w5[k];
		result[k+n_4*3]+=w4[k];
		result[k+n_4*4]+=w3[k];
		result[k+n_4*5]+=w2[k];
		result[k+n_4*6]+=w1[k];
	}
	//std::cout<<"w1:", print_element(w1, sub_size, 1<<logq);//
	//std::cout<<"w2:", print_element(w2, sub_size, 1<<logq);//
	//std::cout<<"w3:", print_element(w3, sub_size, 1<<logq);//
	//std::cout<<"w4:", print_element(w4, sub_size, 1<<logq);//
	//std::cout<<"w5:", print_element(w5, sub_size, 1<<logq);//
	//std::cout<<"w6:", print_element(w6, sub_size, 1<<logq);//
	//std::cout<<"w7:", print_element(w7, sub_size, 1<<logq);//
	//std::cout<<"before reduction:", print_element(result, n*2-1, 1<<logq);//
//#ifdef PROFILER
//	std::cout<<"interpolation:    "<<__rdtsc()-t1<<endl;
//	t1=__rdtsc();
//#endif

	short q_mask=(1<<logq)-1, sign_mask=-(short)anti_cyclic;
	for(int k=0;k<n-1;++k)//reduce mod x^n+1
		ab[k]=ab[k]+result[k]+(result[k+n]^sign_mask)-sign_mask & q_mask;
	//for(int k=n;k<n*2-1;++k)
	//	ab[k-n]=ab[k-n]+result[k-n]+(result[k]^sign_mask)-sign_mask & q_mask;
	//	ab[k-n]=ab[k-n]+result[k-n]-result[k] & q_mask;
	ab[n-1]=ab[n-1]+result[n-1] & q_mask;
	memset(result, 0, result_size*sizeof(short));
	memset(buffer, 0, buf_size*sizeof(short));
	delete[] result, buffer;
//#ifdef PROFILER
//	std::cout<<"reduction:        "<<__rdtsc()-t1<<endl;
//#endif
}
void		saber_bits(short *b, const short *a, int n, int i, int j)
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i mask=_mm256_set1_epi16((1<<j)-1);
	const __m128i sh=_mm_set_epi32(0, 0, 0, i-j);
	for(int kx=0;kx<n;kx+=16)
	{
		__m256i v=_mm256_loadu_si256((__m256i*)(a+kx));
		v=_mm256_sra_epi16(v, sh);
		v=_mm256_and_si256(v, mask);
		_mm256_storeu_si256((__m256i*)(b+kx), v);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i mask=_mm_set1_epi16((1<<j)-1);
	const __m128i sh=_mm_set_epi32(0, 0, 0, i-j);
	for(int kx=0;kx<n;kx+=8)
	{
		__m128i v=_mm_loadu_si128((__m128i*)(a+kx));
		v=_mm_sra_epi16(v, sh);
		v=_mm_and_si128(v, mask);
		_mm_storeu_si128((__m128i*)(b+kx), v);
	}
#else
	for(int kx=0, sh=i-j, mask=(1<<j)-1;kx<n;++kx)
		b[kx]=a[kx]>>sh&mask;
#endif
}
void		saber_bits_h(short *b, int vector_size, int logq, int logp)
{
#if PROCESSOR_ARCH>=AVX2
	const __m256i h=_mm256_set1_epi16(1<<(logq-logp-1));
	const __m256i mask=_mm256_set1_epi16((1<<logp)-1);
	const __m128i sh=_mm_set_epi32(0, 0, 0, logq-logp);
	for(int kx=0;kx<vector_size;kx+=16)
	{
		__m256i v=_mm256_loadu_si256((__m256i*)(b+kx));
		v=_mm256_add_epi16(v, h);
		v=_mm256_sra_epi16(v, sh);
		v=_mm256_and_si256(v, mask);
		_mm256_storeu_si256((__m256i*)(b+kx), v);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i h=_mm_set1_epi16(1<<(logq-logp-1));
	const __m128i mask=_mm_set1_epi16((1<<logp)-1);
	const __m128i sh=_mm_set_epi32(0, 0, 0, logq-logp);
	for(int kx=0;kx<vector_size;kx+=8)
	{
		__m128i v=_mm_loadu_si128((__m128i*)(b+kx));
		v=_mm_add_epi16(v, h);
		v=_mm_sra_epi16(v, sh);
		v=_mm_and_si128(v, mask);
		_mm_storeu_si128((__m128i*)(b+kx), v);
	}
#else
	int h=1<<(logq-logp-1);
	for(int kx=0, sh=logq-logp, mask=(1<<logp)-1;kx<vector_size;++kx)//A.s+h
		b[kx]=(b[kx]+h)>>sh&mask;
#endif
}
void		saber_calculate_b(short *b, const short *A, const short *s, int n, int k, int logq, int logp, bool anti_cyclic)
{
	int vector_size=k*n;
	memset(b, 0, vector_size*sizeof(short));
	for(int ky=0;ky<k;++ky)
		for(int kx=0;kx<k;++kx)//A.s
		//	toom_cook_4way((unsigned short*)A+n*(k*ky+kx), (unsigned short*)s+n*kx, (unsigned short*)b+n*ky, 1<<logq, n);
			multiply_toom_cook4_saber(A+n*(k*ky+kx), s+n*kx, b+n*ky, n, logq, anti_cyclic);
		//	multiply_karatsuba(A+n*(k*ky+kx), s+n*kx, b+n*ky, n, logq, anti_cyclic);
		//	multiply_polynomials_mod_powof2_add(A+n*(k*ky+kx), s+n*kx, b+n*ky, n, logq, anti_cyclic);
	saber_bits_h(b, vector_size, logq, logp);
}
void		saber_calculate_b_dash(short *b_dash, const short *A, const short *s, int n, int k, int logq, int logp, bool anti_cyclic)
{
	int vector_size=k*n;
	memset(b_dash, 0, vector_size*sizeof(short));
	for(int ky=0;ky<k;++ky)
		for(int kx=0;kx<k;++kx)//AT.s
			multiply_toom_cook4_saber(A+n*(k*kx+ky), s+n*kx, b_dash+n*ky, n, logq, anti_cyclic);
		//	multiply_karatsuba(A+n*(k*kx+ky), s+n*kx, b_dash+n*ky, n, logq, anti_cyclic);
		//	multiply_polynomials_mod_powof2_add(A+n*(k*kx+ky), s+n*kx, b_dash+n*ky, n, logq, anti_cyclic);
	saber_bits_h(b_dash, vector_size, logq, logp);
	//int h=1<<(logq-logp-1);
	//for(int kx=0, sh=logq-logp, mask=(1<<logp)-1;kx<vector_size;++kx)//A.s+h
	//	b[kx]=(b[kx]+h)>>sh&mask;
}
void		saber_calculate_v(short *v, const short *b, short *s, int n, int k, int logq, int logp, bool anti_cyclic)
{
	int vector_size=k*n;
	for(int kx=0, mask=(1<<logp)-1;kx<vector_size;++kx)
		s[kx]&=mask;
	memset(v, 0, n*sizeof(short));
	for(int kx=0;kx<k;++kx)
		multiply_toom_cook4_saber(b+n*kx, s+n*kx, v, n, logq, anti_cyclic);
	//	multiply_karatsuba(b+n*kx, s+n*kx, v, n, logq, anti_cyclic);
	//	multiply_polynomials_mod_powof2_add(b+n*kx, s+n*kx, v, n, logq, anti_cyclic);
#if PROCESSOR_ARCH>=AVX2
	const __m256i h1=_mm256_set1_epi16(1<<(logq-logp-1));
	const __m256i mask=_mm256_set1_epi16((1<<logp)-1);
	for(int kx=0;kx<n;kx+=16)
	{
		__m256i vk=_mm256_loadu_si256((__m256i*)(v+kx));
		vk=_mm256_add_epi16(vk, h1);
		vk=_mm256_and_si256(vk, mask);
		_mm256_storeu_si256((__m256i*)(v+kx), vk);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i h1=_mm_set1_epi16(1<<(logq-logp-1));
	const __m128i mask=_mm_set1_epi16((1<<logp)-1);
	for(int kx=0;kx<n;kx+=8)
	{
		__m128i vk=_mm_loadu_si128((__m128i*)(v+kx));
		vk=_mm_add_epi16(vk, h1);
		vk=_mm_and_si128(vk, mask);
		_mm_storeu_si128((__m128i*)(v+kx), vk);
	}
#else
	int h1=1<<(logq-logp-1);
	for(int kx=0, mask=(1<<logp)-1;kx<n;++kx)
		v[kx]=v[kx]+h1&mask;
#endif
}
void		saber_kdf(unsigned char *K, const short *K_dash, int K_dash_size)
{
	int size=K_dash_size>>3;
	memset(K, 0, size);
	for(int kx=0;kx<size;++kx)
		for(int k2=0;k2<8;++k2)
			K[kx]|=K_dash[(kx<<3)+k2]<<k2;
}
struct		Saber_private_key
{
	short *s;//k*n	*4bits = 192 shorts = 384 bytes
};
struct		Saber_public_key
{
	unsigned char *seed_A;//256bit = 32 bytes
	short *b;	//k*n *10bits -> 528 shorts = 1056 bytes
};
struct		Saber_ciphertext
{
	short *b_dash,	//k*n *10bits -> 480 shorts = 960 bytes
		*cm;		//n *(3+1)bits -> 64 shorts = 128 bytes
};
void		saber_generate(Saber_private_key &k_pr, Saber_public_key &k_pu, int n)
{
	//Generate	page 10
//	std::cout<<"\tGeneration:";//
	const int k=3, logq=13, logp=10; const bool anti_cyclic=true;
	const int size=32, A_size=k*k*n, vector_size=k*n;
	k_pu.seed_A=new unsigned char[size];
#ifdef PROFILER
	std::cout<<"KeyGen()\n";
	long long t1=__rdtsc();
#endif
	generate_uniform(size, k_pu.seed_A);
#ifdef PROFILER
	std::cout<<"generate_uniform: "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	short *A=new short[A_size];
	FIPS202_SHAKE128(k_pu.seed_A, size, (unsigned char*)A, A_size*sizeof(short));
//	for(int k2=0;k2<A_size;++k2)A[k2]=1;//
//	for(int kx=0;kx<A_size;kx+=8)A[kx]=10, A[kx+1]=10, A[kx+2]=10, A[kx+3]=10, A[kx+4]=1, A[kx+5]=1, A[kx+6]=1, A[kx+7]=1;
//	for(int kx=0;kx<A_size;kx+=8)A[kx]=1000, A[kx+1]=1000, A[kx+2]=1000, A[kx+3]=1000, A[kx+4]=1, A[kx+5]=1, A[kx+6]=1, A[kx+7]=1;
//	for(int kx=0;kx<A_size;kx+=8)A[kx]=5000, A[kx+1]=5000, A[kx+2]=5000, A[kx+3]=5000, A[kx+4]=400, A[kx+5]=400, A[kx+6]=400, A[kx+7]=400;
//	for(int kx=0;kx<A_size;kx+=8)A[kx]=5972, A[kx+1]=5972, A[kx+2]=5972, A[kx+3]=473, A[kx+4]=473, A[kx+5]=473, A[kx+6]=473, A[kx+7]=473;
//	for(int kx=0;kx<A_size;kx+=8)A[kx]=5972, A[kx+1]=5936, A[kx+2]=5012, A[kx+3]=473, A[kx+4]=6806, A[kx+5]=6569, A[kx+6]=4411, A[kx+7]=1210;
//	std::cout<<"\nA:"; for(int k2=0, k2End=k*k;k2<k2End;++k2)print_element_nnl(A+n*k2, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"generate A:       "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	k_pr.s=new short[vector_size];
	saber_generate_binomial_8(k_pr.s, vector_size);
//	for(int k2=0;k2<vector_size;++k2)s[k2]=1;//
//	std::cout<<"\ns:"; for(int k2=0;k2<k;++k2)print_element_small(k_pr.s+n*k2, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"generate s:       "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	k_pu.b=new short[vector_size];
	saber_calculate_b(k_pu.b, A, k_pr.s, n, k, logq, logp, anti_cyclic);
//	std::cout<<"\nb = A s + 4 >> 3 in Rp:"; for(int k2=0;k2<k;++k2)print_element_nnl(k_pu.b+n*k2, n, 1<<logq);//
	delete[] A;
#ifdef PROFILER
	std::cout<<"b = A s:          "<<__rdtsc()-t1<<endl;
#endif
}
void		saber_encrypt(const char *message, Saber_ciphertext &ct, Saber_public_key const &k_pu, unsigned char *r, int n)
{
	//Encrypt	page 11
//	std::cout<<"\n\n\tEncryption:";//
	const int k=3, logq=13, logp=10, logt=3; const bool anti_cyclic=true;
	const int size=32, A_size=k*k*n, vector_size=k*n;
#ifdef PROFILER
	std::cout<<"Encrypt()\n";
	long long t1=__rdtsc();
#endif
//	const char *message="12345678901234567890123456789012";//256bit
//	std::cout<<"Message:\t"<<message;
	short *m=new short[n];
	for(int kv=0;kv<n;++kv)
		m[kv]=message[kv>>3]>>(kv&7)&1;
#ifdef PROFILER
	std::cout<<"encode(m):        "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	short *A=new short[A_size];
	FIPS202_SHAKE128(k_pu.seed_A, size, (unsigned char*)A, A_size*sizeof(short));
//	for(int k2=0;k2<A_size;++k2)A[k2]=1;//
//	std::cout<<"\nA:"; for(int k2=0, k2End=k*k;k2<k2End;++k2)print_element_nnl(A+n*k2, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"generate A:       "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	short *s_dash=new short[vector_size];
	FIPS202_SHAKE128(r, size, (unsigned char*)s_dash, vector_size*sizeof(short));
	saber_convert_binomial_8(s_dash, vector_size);
//	saber_generate_binomial_8(s_dash, vector_size);
//	memset(s_dash, 0, vector_size*sizeof(short));//
//	for(int k2=0;k2<vector_size;++k2)s_dash[k2]=1;//
//	std::cout<<"\ns':"; for(int k2=0;k2<k;++k2)print_element_small(s_dash+n*k2, n, q);//
#ifdef PROFILER
	std::cout<<"generate s':      "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	ct.b_dash=new short[vector_size];
	saber_calculate_b_dash(ct.b_dash, A, s_dash, n, k, logq, logp, anti_cyclic);//in Rp
//	std::cout<<"\nb' = AT s' + 4 >> 3 in Rp:"; for(int k2=0;k2<k;++k2)print_element_nnl(ct.b_dash+n*k2, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"b' = AT s':       "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	short *v_dash=new short[n];
			//std::cout<<"\t1";//
	saber_calculate_v(v_dash, k_pu.b, s_dash, n, k, logq, logp, anti_cyclic);
//	std::cout<<"\nv' = bT s' in Rp:", print_element_nnl(v_dash, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"v' = bT s':       "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	ct.cm=new short[n];
#if PROCESSOR_ARCH>=AVX2
	const __m128i sh1=_mm_set_epi32(0, 0, 0, logp-1),
		sh2=_mm_set_epi32(0, 0, 0, logp-logt-1);
	for(int kx=0;kx<n;kx+=16)
	{
		__m256i vdk=_mm256_loadu_si256((__m256i*)(v_dash+kx));
		__m256i mk=_mm256_loadu_si256((__m256i*)(m+kx));
		mk=_mm256_sll_epi16(mk, sh1);
		vdk=_mm256_add_epi16(vdk, mk);
		vdk=_mm256_sra_epi16(vdk, sh2);
		_mm256_storeu_si256((__m256i*)(ct.cm+kx), vdk);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i sh1=_mm_set_epi32(0, 0, 0, logp-1),
		sh2=_mm_set_epi32(0, 0, 0, logp-logt-1);
	for(int kx=0;kx<n;kx+=8)
	{
		__m128i vdk=_mm_loadu_si128((__m128i*)(v_dash+kx));
		__m128i mk=_mm_loadu_si128((__m128i*)(m+kx));
		mk=_mm_sll_epi16(mk, sh1);
		vdk=_mm_add_epi16(vdk, mk);
		vdk=_mm_sra_epi16(vdk, sh2);
		_mm_storeu_si128((__m128i*)(ct.cm+kx), vdk);
	}
#else
	for(int kx=0;kx<n;++kx)
		ct.cm[kx]=(v_dash[kx]+(m[kx]<<(logp-1)))>>(logp-logt-1);//in R2t
#endif
//	std::cout<<"\ncm = v' + (m<<ep-1) >> ep-et-1 in Rp:", print_element_nnl(ct.cm, n, 1<<logq);//
	delete[] m, A, s_dash, v_dash;
#ifdef PROFILER
	std::cout<<"calculate cm:     "<<__rdtsc()-t1<<endl;
#endif
}
void		saber_decrypt(Saber_ciphertext const &ct, char *message2, Saber_private_key const &k_pr, Saber_public_key const &k_pu, int n)
{
	//Decrypt	page 11
//	std::cout<<"\n\n\tDecryption:";//
	const int k=3, logq=13, logp=10, logt=3; const bool anti_cyclic=true;
	const int size=32, A_size=k*k*n, vector_size=k*n;
#ifdef PROFILER
	std::cout<<"Decrypt()\n";
	long long t1=__rdtsc();
#endif
	short *v=new short[n];
	saber_calculate_v(v, ct.b_dash, k_pr.s, n, k, logq, logp, anti_cyclic);
//	std::cout<<"\nv = b'T s in Rp:", print_element_nnl(v, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"v = b'T s:        "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	short *m_dash=new short[n];
#if PROCESSOR_ARCH>=AVX2
	const __m256i h2=_mm256_set1_epi16((1<<(logp-2))-(1<<(logp-logt-2)));
	const __m128i sh1=_mm_set_epi32(0, 0, 0, logp-logt-1);
	const __m128i sh2=_mm_set_epi32(0, 0, 0, logp-1);
	const __m256i one=_mm256_set1_epi16(1);
	for(int kx=0;kx<n;kx+=16)
	{
		__m256i vk=_mm256_loadu_si256((__m256i*)(v+kx));
		__m256i cmk=_mm256_loadu_si256((__m256i*)(ct.cm+kx));
		cmk=_mm256_sll_epi16(cmk, sh1);
		vk=_mm256_sub_epi16(vk, cmk);
		vk=_mm256_add_epi16(vk, h2);
		vk=_mm256_sra_epi16(vk, sh2);
		vk=_mm256_and_si256(vk, one);
		_mm256_storeu_si256((__m256i*)(m_dash+kx), vk);
	}
#elif PROCESSOR_ARCH>=SSE2
	const __m128i h2=_mm_set1_epi16((1<<(logp-2))-(1<<(logp-logt-2))),
		sh1=_mm_set_epi32(0, 0, 0, logp-logt-1),
		sh2=_mm_set_epi32(0, 0, 0, logp-1),
		one=_mm_set1_epi16(1);
	for(int kx=0;kx<n;kx+=8)
	{
		__m128i vk=_mm_loadu_si128((__m128i*)(v+kx));
		__m128i cmk=_mm_loadu_si128((__m128i*)(ct.cm+kx));
		cmk=_mm_sll_epi16(cmk, sh1);
		vk=_mm_sub_epi16(vk, cmk);
		vk=_mm_add_epi16(vk, h2);
		vk=_mm_sra_epi16(vk, sh2);
		vk=_mm_and_si128(vk, one);
		_mm_storeu_si128((__m128i*)(m_dash+kx), vk);
	}
#else
	short h2=(1<<(logp-2))-(1<<(logp-logt-2)), log_h=logp-logt-1;
	for(int kx=0;kx<n;++kx)
		m_dash[kx]=(v[kx]-(ct.cm[kx]<<log_h)+h2)>>(logp-1)&1;
#endif
//	std::cout<<"\nm' = v - (cm<<ep-et-1) + 4 >> ep in Rp:", print_element_small(m_dash, n, 1<<logq);//
#ifdef PROFILER
	std::cout<<"calculate m':     "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
//	char message2[(n>>3)+1]={0};
	saber_kdf((unsigned char*)message2, m_dash, n);
//	std::cout<<"\nDecryption:\t"<<message2<<endl;
	delete[] v, m_dash;
#ifdef PROFILER
	std::cout<<"kdf:              "<<__rdtsc()-t1<<endl;
#endif
}
void		saber_cca_encapsulate(Saber_public_key const &pu_k, unsigned char *K, Saber_ciphertext &ct, int n)
{
	const int size=32, k=3, vector_size=k*n;

	unsigned char *message=new unsigned char[size];
	generate_uniform(size, message);
//	std::cout<<"message:\t", print_buffer(message, size);//

	int pum_size=vector_size*sizeof(short)+size*2;
	unsigned char *pum_buffer=new unsigned char[pum_size];
	memcpy(pum_buffer, pu_k.b, vector_size*sizeof(short));
	memcpy(pum_buffer+vector_size*sizeof(short), pu_k.seed_A, size);
	memcpy(pum_buffer+vector_size*sizeof(short)+size, message, size);
//	std::cout<<"pu_k.b:\t", print_buffer(pu_k.b, vector_size*sizeof(short));//
//	std::cout<<"pum_buffer:\t", print_buffer(pum_buffer, pum_size);//

	int kr_size=size<<1;
	unsigned char *kr_buffer=new unsigned char[kr_size],
		*key=kr_buffer, *r=kr_buffer+size;
	FIPS202_SHAKE128(pum_buffer, pum_size, kr_buffer, kr_size);
	delete[] pum_buffer;
//	std::cout<<"key:\t", print_buffer(key, size);//
//	std::cout<<"r:\t", print_buffer(r, size);//

	saber_encrypt((char*)message, ct, pu_k, r, n);
	delete[] message;

	int kc_size=size+(vector_size+n)*sizeof(short);
	unsigned char *kc_buffer=new unsigned char[kc_size];
	memcpy(kc_buffer, key, size);
	memcpy(kc_buffer+size, ct.b_dash, vector_size*sizeof(short));
	memcpy(kc_buffer+size+vector_size*sizeof(short), ct.cm, n*sizeof(short));
	delete[] kr_buffer;
//	std::cout<<"kc_buffer:\t", print_buffer(kc_buffer, kc_size);//

	FIPS202_SHAKE128(kc_buffer, kc_size, K, size);
	delete[] kc_buffer;
}
bool		saber_cca_decapsulate(Saber_ciphertext const &ct, unsigned char *K2, Saber_private_key const &pr_k, Saber_public_key const &pu_k, int n)
{
	const int size=32, k=3, vector_size=k*n;

	unsigned char *message2=new unsigned char[size];
	saber_decrypt(ct, (char*)message2, pr_k, pu_k, n);
//	std::cout<<"message2:\t", print_buffer(message2, size);//
	
	int pum2_size=vector_size*sizeof(short)+size*2;
	unsigned char *pum2_buffer=new unsigned char[pum2_size];
	memcpy(pum2_buffer, pu_k.b, vector_size*sizeof(short));
	memcpy(pum2_buffer+vector_size*sizeof(short), pu_k.seed_A, size);
	memcpy(pum2_buffer+vector_size*sizeof(short)+size, message2, size);
//	std::cout<<"pu_k.b:\t", print_buffer(pu_k.b, vector_size*sizeof(short));//
//	std::cout<<"pum_buffer:\t", print_buffer(pum2_buffer, pum2_size);//
	
	int kr2_size=size<<1;
	unsigned char *kr2_buffer=new unsigned char[kr2_size],
		*key2=kr2_buffer, *r2=kr2_buffer+size;
	FIPS202_SHAKE128(pum2_buffer, pum2_size, kr2_buffer, kr2_size);
	delete[] pum2_buffer;
	//std::cout<<"key2:\t", print_buffer(key2, size);//
	//std::cout<<"r2:\t", print_buffer(r2, size);//

	Saber_ciphertext ct2;
	saber_encrypt((char*)message2, ct2, pu_k, r2, n);
	delete[] message2;

	int success=!memcmp(ct.b_dash, ct2.b_dash, vector_size*sizeof(short));
	success&=!memcmp(ct.cm, ct2.cm, n*sizeof(short));
	//std::cout<<"ct.b_dash:\t", print_buffer(ct.b_dash, vector_size*sizeof(short));//
	//std::cout<<"ct2.b_dash:\t", print_buffer(ct2.b_dash, vector_size*sizeof(short));//
	//std::cout<<"ct.cm:\t", print_buffer(ct.cm, n*sizeof(short));//
	//std::cout<<"ct2.cm:\t", print_buffer(ct2.cm, n*sizeof(short));//
	
	int kc_size=size+(vector_size+n)*sizeof(short);
	unsigned char *kc_buffer=new unsigned char[kc_size];
	if(success)
		memcpy(kc_buffer, key2, size);
	else
		generate_uniform(size, key2);
	memcpy(kc_buffer+size, ct.b_dash, vector_size*sizeof(short));
	memcpy(kc_buffer+size+vector_size*sizeof(short), ct.cm, n*sizeof(short));
	delete[] kr2_buffer;
	//std::cout<<"kc_buffer:\t", print_buffer(kc_buffer, kc_size);//
	
	FIPS202_SHAKE128(kc_buffer, kc_size, K2, size);
	delete[] kc_buffer;

	return success!=0;
}

struct		R5_Parameters
{
	//Main parameters
	unsigned char tau;//The variant for creating A
	unsigned short d;//Dimension parameter d
	unsigned short n;//Dimension parameter n
	unsigned short h;//Hamming weight parameter h
	unsigned char q_bits;//The number of q bits
	unsigned char p_bits;//Number of p bits
	unsigned char t_bits;//Number of t bits
	unsigned short n_bar;//Dimension parameter
	unsigned short m_bar;//Dimension parameter
	unsigned char b_bits;//Number of extracted bits per ciphertext symbol (parameter b in bits)
	unsigned char kappa_bytes;//The size of shared secret, in bytes, also used for the size of seeds
	unsigned char f;//Number of bit errors corrected, parameter f
	unsigned char xe;//Number of bits for error correction
	//Derived parameters
	unsigned short kappa;//Parameter kappa, the size of shared secret, in bits, also used for the size of seeds
	unsigned short k;//Dimension parameter k = d/n
	unsigned q;//Parameter q
	unsigned short p;//Parameter p = 2^p_bits
	unsigned short mu;//Parameter mu = (kappa + xe) / B
	//Rounding constants
	unsigned short z_bits, h1, h2, h3;
	//Derived NIST parameters
	unsigned pk_size;//Size of the public key, in bytes
	unsigned short ct_size;//Size of the cipher text, in bytes
};
unsigned r5_A_fixed_size=0;
unsigned short *r5_A_fixed=0;
void		r5_create_A_random(unsigned short *A, const unsigned char *seed, R5_Parameters const &p)
{
	unsigned num_elements;
	switch(p.tau)
	{
	case 0:
	case 1:
		num_elements=p.d*p.k;
		break;
	case 2:
		num_elements=p.q;
		break;
	}
	//drbg_start(seed);
	//drbg_get(A, num_elements*sizeof(unsigned short));
	if(p.kappa_bytes>16)
		FIPS202_SHAKE256(seed, p.kappa_bytes, (unsigned char*)A, num_elements*sizeof(unsigned short));
	else
		FIPS202_SHAKE128(seed, p.kappa_bytes, (unsigned char*)A, num_elements*sizeof(unsigned short));

	for(unsigned i=0;i<num_elements;++i)
		A[i]&=p.q-1;
//	memset(A, 0, num_elements*sizeof(unsigned short));//
	//for(unsigned i=0;i<num_elements;++i)//
	//	A[i]=10;//
}
void		r5_create_A_fixed(const unsigned char *seed, R5_Parameters const &p)
{
    r5_A_fixed_size=p.d*p.k;

    //(Re)allocate space for A_fixed
    r5_A_fixed=(unsigned short*)realloc(r5_A_fixed, r5_A_fixed_size*sizeof(unsigned short));

    //Create A_fixed randomly
    r5_create_A_random(r5_A_fixed, seed, p);

    //Make all elements mod q
    for (unsigned i=0;i<r5_A_fixed_size;++i)
        r5_A_fixed[i]&=p.q-1;
}
void		r5_create_A(unsigned short *A, const unsigned char *sigma, R5_Parameters const &p)
{
	unsigned short *A_master=0;
	unsigned A_permutation=0;
	const unsigned short els_row=p.k*p.n;
	switch(p.tau)
	{
	case 0:
		r5_create_A_random(A, sigma, p);
		break;
/*	case 1:
		A_master=A_fixed;
		break;
	case 2:
		A_master=(unsigned short*)malloc((p.q+p.d)*sizeof(unsigned short));
		r5_create_A_random(A_master, sigma, p);
		memcpy(A_master+p.q, A_master, p.d*sizeof(unsigned short));
		break;//*/
	}
/*	if(p.tau==1||p.tau==2)
	{
		A_permutation=(unsigned*)malloc(p.k*sizeof(unsigned));
		if(p.tau==1)
		{
		}
	}//*/
}
void		r5_create_secret_vector_idx(short *vector_idx, const short size, const unsigned short h, const unsigned short *idx_buffer)
{
/*	short *per=(short*)malloc(size*sizeof(short));
	for(short i=0;i<size;++i)
		per[i]=i;
	//	per[i]=(i+2)%size;//
	for(short i=size-1;i>1;--i)
	{
		unsigned short idx=idx_buffer[i]%i;
		short temp=per[i]; per[i]=per[idx], per[idx]=temp;//Fisher-Yates shuffle
	}
	memcpy(vector_idx, per, h*sizeof(short));
	free(per);//*/
	
	short *vector=(short*)malloc(size*sizeof(short));
	memset(vector, -1, (h>>1)*sizeof(short));
	for(unsigned i=h>>1;i<h;++i)
		vector[i]=1;
	memset(vector+h, 0, (size-h)*sizeof(short));
//	print_element(vector, size, 8192);
	for(short i=size-1;i>1;--i)
	{
		unsigned short idx=idx_buffer[i]%i;
		short temp=vector[i]; vector[i]=vector[idx], vector[idx]=temp;//Fisher-Yates shuffle
	}
//	print_element(vector, size, 8192);
	for(int i=0, ko=0, km=1;i<size;++i)
		if(vector[i]==1)
			vector_idx[ko]=i, ko+=2;
		else if(vector[i]==-1)
			vector_idx[km]=i, km+=2;
	free(vector);//*/
}
void		r5_create_secret_vector(short *vector, const short size, const unsigned short h, const unsigned short *idx_buffer)
//void		r5_create_secret_vector(short *vector, const short size, const unsigned short h, const unsigned char *rand_buffer, unsigned rb_size)
{
	memset(vector, -1, (h>>1)*sizeof(short));
	for(unsigned i=h>>1;i<h;++i)
		vector[i]=1;
	memset(vector+h, 0, (size-h)*sizeof(short));
	for(short i=size-1;i>1;--i)
	{
		unsigned short idx=idx_buffer[i]%i;
		short temp=vector[i]; vector[i]=vector[idx], vector[idx]=temp;//Fisher-Yates shuffle
	}

	//int n1=0, n_1=0;
	//for(int i=0;i<size;++i)
	//	n1+=vector[i]==1, n_1+=vector[i]==-1;
	//std::cout<<"n ones: "<<n1<<", n minus ones "<<n_1<<endl;
//	memset(vector, 0, size*sizeof(short));//

/*	memset(vector, 0, size*sizeof(short));
	for(unsigned i=0, j=0;i<h;++i)
	{
		unsigned short idx;
		do
		{
			idx=rand_buffer[j];
			++j;
			if(j>=h)			//not constant-time
			{
				FIPS202_SHAKE128(rand_buffer, rb_size, rand_buffer, rb_size);
				j=0;
			}
		//	CryptGenRandom(hProv, sizeof(unsigned short), &idx), idx%=size;
		//	idx=rand()%size;//
		}
		while(vector[idx]!=0);
		vector[idx]=i&1?-1:1;
	}//*/
}
void		r5_create_S_T(short *S_T, const unsigned char *sk, R5_Parameters const &p)
{
/*	const unsigned short size=p.k*p.n;
#ifdef R5_USE_IDX
	const unsigned short size_idx=p.k*p.h;
#endif
	unsigned char *v=(unsigned char*)malloc(size);
	memset(v, 0, size);
	drbg_start(sk);	//p.kappa_bytes >= 16
	for(unsigned i=0;i<p.n_bar;++i)
	{
		short *s_idx=S_T+i*size_idx;
		for(unsigned j=0;j<size_idx;++j)
		{
		//	long long t1=__rdtsc();//
			unsigned short idx, mask=(1<<(log_2(p.n)+1))-1;
			do
			{
				drbg_get(&idx, sizeof(unsigned short));
				idx&=mask;
			}
			while(idx>size||v[idx]);
			v[idx]=1;
			s_idx[j]=idx;
		//	std::cout<<__rdtsc()-t1<<endl;//
		}
	}
	free(v);//*/

	const unsigned short size=p.k*p.n;
#ifdef R5_USE_IDX
	const unsigned short size_idx=p.k*p.h;
#endif
	unsigned short *idx_buffer=(unsigned short*)malloc(size*sizeof(unsigned short));
	FIPS202_SHAKE128(sk, p.kappa_bytes, (unsigned char*)idx_buffer, size*sizeof(unsigned short));
	for(unsigned i=0;i<p.n_bar;++i)
	{
#ifdef R5_USE_IDX
		r5_create_secret_vector_idx(S_T+i*size_idx, size, p.h, idx_buffer);
#else
		r5_create_secret_vector(S_T+i*size, size, p.h, idx_buffer);
#endif
		if(i+1<p.n_bar)
			FIPS202_SHAKE128((unsigned char*)idx_buffer, size*sizeof(unsigned short), (unsigned char*)idx_buffer, size*sizeof(unsigned short));
	}//*/
//	memset(S_T, 0, p.h*sizeof(short));//

/*	unsigned rb_size=h*sizeof(short);
	short *rand_buffer=(short*)malloc(rb_size);
	FIPS202_SHAKE128(seed, seed_size, temp, h*sizeof(short));

	for(unsigned i=0;i<p.n_bar;++i)
	{
		r5_create_secret_vector(S_T+i*size, size, p.h, rand_buffer, rb_size);
		FIPS202_SHAKE128(rand_buffer, rb_size, rand_buffer, rb_size);
	}

	free(temp);//*/
}
void		r5_transpose_matrix(unsigned short *matrix_T, const unsigned short *matrix, const unsigned rows, const unsigned cols, const unsigned els)
{
	for(unsigned kr=0;kr<rows;++kr)
		for(unsigned kc=0;kc<cols;++kc)
			memcpy(matrix_T+kc*rows*els+kr*els, matrix+kr*cols*els+kc*els, els*sizeof(short));
			//for(unsigned ke=0;ke<els;++ke)
			//	matrix_T[kc*rows*els+kr*els+ke]=matrix[kr*cols*els+kc*els+ke];
}
void		r5_mult_poly_ntru_idx(unsigned short *result, const short *pol_a, const short *pol_b_idx, const unsigned len, const unsigned h, const unsigned mod)
//void		r5_mult_poly_ntru_idx(unsigned short *result, const short *pol_a, const short (*pol_b_idx)[2], const unsigned len, const unsigned h, const unsigned mod)
{
#if PROCESSOR_ARCH>=AVX2
	short *pol_a2=(short*)malloc(2*len*sizeof(short));
	memcpy(pol_a2, pol_a, len*sizeof(short));
	memcpy(pol_a2+len, pol_a, len*sizeof(short));
	for(unsigned i=0;i<h;i+=2)
	{
		unsigned idx_add=pol_b_idx[i], idx_sub=pol_b_idx[i+1];
		unsigned j=0;
		for(;j+16<len;j+=16)
		{
			__m256i mr=_mm256_loadu_si256((__m256i*)(result+j));
			__m256i pa=_mm256_loadu_si256((__m256i*)(pol_a2+idx_add+j));
			__m256i ps=_mm256_loadu_si256((__m256i*)(pol_a2+idx_sub+j));
			mr=_mm256_add_epi16(mr, pa);
			mr=_mm256_sub_epi16(mr, ps);
			_mm256_storeu_si256((__m256i*)(result+j), mr);
		}
		for(;j<len;++j)
			result[j]+=pol_a2[idx_add+j]-pol_a2[idx_sub+j];
	}
	free(pol_a2);
	_mm_empty();
#elif PROCESSOR_ARCH>=SSE2
	short *pol_a2=(short*)malloc(2*len*sizeof(short));
	memcpy(pol_a2, pol_a, len*sizeof(short));
	memcpy(pol_a2+len, pol_a, len*sizeof(short));
	for(unsigned i=0;i<h;i+=2)
	{
		unsigned idx_add=pol_b_idx[i], idx_sub=pol_b_idx[i+1];
		unsigned j=0;
		for(;j+8<len;j+=8)
		{
			__m128i mr=_mm_loadu_si128((__m128i*)(result+j));
			__m128i pa=_mm_loadu_si128((__m128i*)(pol_a2+idx_add+j));
			__m128i ps=_mm_loadu_si128((__m128i*)(pol_a2+idx_sub+j));
			mr=_mm_add_epi16(mr, pa);
			mr=_mm_sub_epi16(mr, ps);
			_mm_storeu_si128((__m128i*)(result+j), mr);
		}
		for(;j<len;++j)
			result[j]+=pol_a2[idx_add+j]-pol_a2[idx_sub+j];
	}
	free(pol_a2);
	_mm_empty();
#else
	short *pol_a2=(short*)malloc(2*len*sizeof(short));
	memcpy(pol_a2, pol_a, len*sizeof(short));
	memcpy(pol_a2+len, pol_a, len*sizeof(short));
	for(unsigned i=0;i<h;i+=2)
	{
		unsigned idx_add=pol_b_idx[i], idx_sub=pol_b_idx[i+1];
		unsigned j=0;
		for(;j+4<len;j+=4)
		{
			(__m64&)result[j]=_mm_add_pi16((__m64&)result[j], (__m64&)pol_a2[idx_add+j]);
			(__m64&)result[j]=_mm_sub_pi16((__m64&)result[j], (__m64&)pol_a2[idx_sub+j]);
		}
		for(;j<len;++j)
			result[j]+=pol_a2[idx_add+j]-pol_a2[idx_sub+j];
	}
	free(pol_a2);

	//short *pol_a2=(short*)malloc(2*len*sizeof(short));
	//memcpy(pol_a2, pol_a, len*sizeof(short));
	//memcpy(pol_a2+len, pol_a, len*sizeof(short));
	//for(unsigned i=0;i<h;i+=2)
	//{
	//	unsigned idx_add=pol_b_idx[i], idx_sub=pol_b_idx[i+1];
	//	for(unsigned j=0;j<len;++j)
	//		result[j]+=pol_a2[idx_add+j]-pol_a2[idx_sub+j];
	//}
	//free(pol_a2);

	//for(unsigned i=0;i<h;i+=2)
	//{
	//	unsigned idx_add=pol_b_idx[i], idx_sub=pol_b_idx[i+1];
	//	for(unsigned j=0;j<len;++j)
	//		result[j]+=pol_a[(idx_add+j)%len]-pol_a[(idx_sub+j)%len];
	//}

	//for(unsigned i=0;i<h;i+=2)
	//{
	//	unsigned id=pol_b_idx[i];
	//	for(unsigned j=id;j<len;++j)
	//		result[j]+=pol_a[j-id];
	//	for(unsigned j=0;j<id;++j)//DEBUG
	//		result[j]+=pol_a[i-id+len];
	//	id=pol_b_idx[i+1];
	//	for(unsigned j=id;j<len;++j)
	//		result[j]-=pol_a[j-id];
	//	for(unsigned j=0;j<id;++j)//DEBUG
	//		result[j]-=pol_a[i-id+len];
	//}

/*	for(unsigned i=0;i<h;i+=2)
	{
		unsigned id=pol_b_idx[i];
		for(unsigned j=0;id+j<len;++j)
			result[id+j]+=pol_a[j];
		for(unsigned j=len-id;j<len;++j)
			result[id+j-len]+=pol_a[j];
		id=pol_b_idx[i+1];
		for(unsigned j=0;id+j<len;++j)
			result[id+j]-=pol_a[j];
		for(unsigned j=len-id;j<len;++j)
			result[id+j-len]-=pol_a[j];
	}//*/

	//for(unsigned i=0;i<h;i+=2)
	//{
	//	for(unsigned j=0;j<len;++j)
	//		result[(pol_b_idx[i  ]+j)%len]+=pol_a[j];
	//	for(unsigned j=0;j<len;++j)
	//		result[(pol_b_idx[i+1]+j)%len]-=pol_a[j];
	//}
//#endif
#endif
}
void		r5_mult_poly_ntru(unsigned short *result, const short *pol_a, const short *ternary_pol_b, const unsigned len, const unsigned mod)
{
	memset(result, 0, len*sizeof(unsigned short));

	const int h=384;
	short *ones=new short[h];
	memset(ones, 0, h*sizeof(short));
	for(unsigned i=0, ko=0, km=1;i<len;++i)
	{
		if(ternary_pol_b[i]==1)
			ones[ko]=i, ko+=2;
		else if(ternary_pol_b[i]==-1)
			ones[km]=i, km+=2;
	}
	r5_mult_poly_ntru_idx(result, pol_a, ones, len, h, mod);
	//for(unsigned i=0;i<h;i+=2)
	//{
	//	int ko=ones[i], km=ones[i+1];
	//	for(unsigned j=0;j<len;++j)
	//		result[(ko+j)%len]+=pol_a[j];
	//	for(unsigned j=0;j<len;++j)
	//		result[(km+j)%len]-=pol_a[j];
	//}
	delete[] ones;//*/

/*	unsigned o_size=0, m_size=0;
	unsigned *ones=0, *mones=0;
	for(unsigned i=0;i<len;++i)
	{
		if(ternary_pol_b[i]==1)
			++o_size, ones=(unsigned*)realloc(ones, o_size*sizeof(unsigned)), ones[o_size-1]=i;
		else if(ternary_pol_b[i]==-1)
			++m_size, mones=(unsigned*)realloc(mones, m_size*sizeof(unsigned)), mones[m_size-1]=i;
	}
	for(unsigned i=0;i<o_size;++i)
		for(unsigned j=0;j<len;++j)
			result[(ones[i  ]+j)%len]+=pol_a[j];
	for(unsigned i=0;i<m_size;++i)
		for(unsigned j=0;j<len;++j)
			result[(mones[i  ]+j)%len]-=pol_a[j];
	free(ones), free(mones);//*/

/*	for(unsigned i=0;i<len;++i)
	{
		if(ternary_pol_b[i]==1)
			for(unsigned j=0;j<len;++j)
				result[(i+j)%len]+=pol_a[j];
		else if(ternary_pol_b[i]==-1)
			for(unsigned j=0;j<len;++j)
				result[(i+j)%len]-=pol_a[j];
	}//*/

//	multiply_polynomials_mod_powof2_add(pol_a, ternary_pol_b, (short*)result, len, log_2(mod), false);

/*	for(unsigned i=0;i<len;++i)
	{
		for(unsigned j=0;j<len;++j)
		{
			unsigned deg=(i+j)%len;
			result[deg]=result[deg]+pol_a[i]*ternary_pol_b[j]&mod-1;
		}
	}//*/
}
void		r5_mult_poly(unsigned short *result, const short *pol_a, const short *ternary_pol_b, const unsigned len, const unsigned h, const unsigned mod, const int isXi)
{
#ifdef PROFILER
	std::cout<<"mult_poly:\n";
	long long t1=__rdtsc();
#endif
	unsigned short *ntru_a=(unsigned short*)malloc((len+1)*sizeof(unsigned short));
#ifndef R5_USE_IDX
	short *ntru_b=(short*)malloc((len+1)*sizeof(short));
#endif
	unsigned short *ntru_res=(unsigned short*)malloc((len+1)*sizeof(unsigned short));
	if(isXi)
	{
		memcpy(ntru_a, pol_a, len*sizeof(unsigned short));
		ntru_a[len] = 0;
	}
	else//lift_poly		multiply by X-1
	{
		ntru_a[0]=-pol_a[0]&mod-1;
		for(unsigned i=1;i<len;++i)
			ntru_a[i]=pol_a[i-1]-pol_a[i]&mod-1;
		ntru_a[len]=pol_a[len-1]&mod-1;
	}
#ifdef PROFILER
	std::cout<<"lift_poly:         "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
#ifdef _DEBUG
	#ifdef R5_USE_IDX
	std::cout<<"b_idx:\t", print_element(ternary_pol_b, h, mod);//
	#else
	std::cout<<"b_idx:\t", print_element(ternary_pol_b, len, mod);//
	#endif
	std::cout<<"ntru_a:\t", print_element((short*)ntru_a, len+1, mod);//
#endif

	memset(ntru_res, 0, (len+1)*sizeof(unsigned short));
#ifdef R5_USE_IDX
	r5_mult_poly_ntru_idx(ntru_res, (short*)ntru_a, ternary_pol_b, len+1, h, mod);
#else
	memcpy(ntru_b, ternary_pol_b, len*sizeof(short));
	ntru_b[len]=0;
	r5_mult_poly_ntru(ntru_res, (short*)ntru_a, ntru_b, len+1, mod);
#endif
#ifdef PROFILER
	std::cout<<"mul:               "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	
#ifdef _DEBUG
	std::cout<<"ntru_res:\t", print_element((short*)ntru_res, len+1, mod);//
#endif
	if(isXi)
		memcpy(result, ntru_res+1, len*sizeof(unsigned short));
	else//unlift_poly		divide by X-1
	{
		result[0]=-ntru_res[0]&mod-1;
		for(unsigned i=1;i<len;++i)
			result[i]=result[i-1]-ntru_res[i]&mod-1;
	}
#ifdef PROFILER
	std::cout<<"unlift_poly:       "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
//#ifdef _DEBUG
//	std::cout<<"result:\t", print_element((short*)result, len, mod);//
//#endif

	free(ntru_a);
#ifndef R5_USE_IDX
	free(ntru_b);
#endif
	free(ntru_res);
}
void		r5_mult_matrix(unsigned short *result, const short *left, const unsigned l_rows, const short *right, const unsigned l_cols_r_rows, const unsigned r_cols, const unsigned els, const unsigned h, const unsigned mod, const int isXi)
{
	memset(result, 0, l_rows*r_cols*els*sizeof(unsigned short));
	
	unsigned short *temp_poly=(unsigned short*)malloc(els*sizeof(unsigned short));
	for(unsigned kr=0;kr<l_rows;++kr)
		for(unsigned kc=0;kc<r_cols;++kc)
			for(unsigned ke=0;ke<l_cols_r_rows;++ke)
			{
				r5_mult_poly(temp_poly, left+kr*l_cols_r_rows*els+ke*els, right+ke*r_cols*els+kc*els, els, h, mod, isXi);
				unsigned short *rk=result+kr*r_cols*els+kc*els;
				for(unsigned i=0;i<els;++i)
					rk[i]=rk[i]+temp_poly[i]&mod-1;
			}
	free(temp_poly);
}
void		r5_mult_matrix_swapped(unsigned short *result, const short *left, const unsigned l_rows, const short *right, const unsigned l_cols_r_rows, const unsigned r_cols, const unsigned els, const unsigned h, const unsigned mod, const int isXi)
{
	memset(result, 0, l_rows*r_cols*els*sizeof(unsigned short));
	
	unsigned short *temp_poly=(unsigned short*)malloc(els*sizeof(unsigned short));
	for(unsigned kr=0;kr<l_rows;++kr)
		for(unsigned kc=0;kc<r_cols;++kc)
			for(unsigned ke=0;ke<l_cols_r_rows;++ke)
			{
				r5_mult_poly(temp_poly, right+ke*r_cols*els+kc*els, left+kr*l_cols_r_rows*els+ke*els, els, h, mod, isXi);
				unsigned short *rk=result+kr*r_cols*els+kc*els;
				for(unsigned i=0;i<els;++i)
					rk[i]=rk[i]+temp_poly[i]&mod-1;
			}
	free(temp_poly);
}
void		r5_round_matrix(unsigned short *matrix, const unsigned len, const unsigned els, const unsigned short a, const unsigned short b, const unsigned short rounding_constant)
{
	unsigned short b_mask=(1<<b)-1;
	for(unsigned i=0;i<len*els;++i)
		matrix[i]=(matrix[i]+rounding_constant)>>(a-b)&b_mask;
}
unsigned	r5_pack(unsigned char *packed, const unsigned short *m, const unsigned els, const unsigned char nr_bits)
{
	const unsigned packed_len=(els*nr_bits+7)>>3;
	const unsigned short mask=(1<<nr_bits)-1;
	unsigned i;
	unsigned short val;
	unsigned bits_done = 0;
	unsigned idx;
	unsigned bit_idx;

	memset(packed, 0, packed_len);
	if(nr_bits==8)
		for(i=0;i<els;++i)
			packed[i]=(unsigned char)m[i];
	else
		for(i=0;i<els;++i)
		{
			idx=bits_done>>3, bit_idx=bits_done&7;
			val=m[i]&mask;
			packed[idx]|=val<<bit_idx;
			if(bit_idx+nr_bits>8)
			{
				//Spill over to next packed byte
				packed[idx+1]|=val>>(8-bit_idx);
				if(bit_idx+nr_bits>16)
					//Spill over to next packed byte
					packed[idx+2]|=val>>(16-bit_idx);
			}
			bits_done+=nr_bits;
		}
	return packed_len;
}
unsigned	r5_unpack(unsigned short *m, const unsigned char *packed, const unsigned els, const unsigned char nr_bits)
{
    const unsigned unpacked_len=(els*nr_bits+7)/8;
    unsigned i;
    unsigned short val;
    unsigned bits_done=0;
    unsigned idx;
    unsigned bit_idx;
    unsigned short bitmask=(1<<nr_bits)-1;
    if(nr_bits==8)
        for(i=0;i<els;++i)
            m[i]=packed[i];
    else
        for(i=0;i<els;++i)
		{
            idx=bits_done>>3;
            bit_idx=bits_done&7;
            val=packed[idx]>>bit_idx;
            if(bit_idx+nr_bits>8)
			{
                //Get spill over from next packed byte
                val|=packed[idx+1]<<(8-bit_idx);
                if(bit_idx+nr_bits>16)
                    //Get spill over from next packed byte
                    val|=packed[idx+2]<<(16-bit_idx);
            }
            m[i]=val&bitmask;
            bits_done+=nr_bits;
        }
    return unpacked_len;
}
void		r5_pack_q_p(unsigned char *pv, const unsigned short *vq, const unsigned short rounding_constant, R5_Parameters const &p)
{
	if(p.p_bits==8)
	{
		for(unsigned i=0;i<p.n;i++)
			pv[i]=(vq[i]+rounding_constant)>>(p.q_bits-p.p_bits)&(p.p-1);
	}
	else
	{
		const unsigned ndp_size=(p.n*p.p_bits+7)>>3;
		memset(pv, 0, ndp_size);
		for(unsigned i=0, j=0;i<p.n;++i)
		{
			unsigned short t=(vq[i]+rounding_constant)>>(p.q_bits-p.p_bits)&(p.p-1);
			pv[j>>3]|=t<<(j&7);//pack p bits
			if((j&7)+p.p_bits>8)
				pv[(j>>3)+1]|=t>>(8-(j&7));
			j+=p.p_bits;
		}
	}
}
void		r5_unpack_p(unsigned short *vp, const unsigned char *pv, R5_Parameters const &p)
{
	if(p.p_bits==8)
		memcpy(vp, pv, p.n);
	else
	{
		for (unsigned i=0, j=0;i<p.n;++i)
		{
			unsigned short t=pv[j>>3]>>(j&7); // unpack p bits
			if((j&7)+p.p_bits>8)
				t|=(unsigned short)pv[(j >> 3) + 1]<<(8-(j&7));
			vp[i]=t&(p.p-1);
			j+=p.p_bits;
		}
	}
}
const unsigned xef_reg[5][3][10] = {
	{
		{ 11, 13 }, // XE1-24
		{ 13, 15 }, // XE1-28
		{ 16, 16 }  // XE1-32
	}, {
		{ 11, 13, 14, 15 }, // XE2-53
		{ 13, 15, 16, 17 }, // XE2-61
		{ 16, 16, 17, 19 }  // XE2-68
	}, {
		{ 11, 13, 15, 16, 17, 19 }, // XE3-91
		{ 13, 15, 16, 17, 19, 23 }, // XE3-103
		{ 16, 16, 17, 19, 21, 23 }  // XE3-112
	}, {
		{ 11, 13, 16, 17, 19, 21, 23, 29 }, // XE4-149
		{ 13, 15, 16, 17, 19, 23, 29, 31 }, // XE4-163
		{ 16, 16, 17, 19, 21, 23, 25, 29 }  // XE4-166
	}, {
		{ 16, 11, 13, 16, 17, 19, 21, 23, 25, 29 }, // XE5-190
		{ 24, 13, 16, 17, 19, 21, 23, 25, 29, 31 }, // XE5-218
		{ 16, 16, 17, 19, 21, 23, 25, 29, 31, 37 }  // XE5-234
	}
};
unsigned	xef_compute(void *block, unsigned len, unsigned f)
{
	unsigned char *v=(unsigned char*)block;
	unsigned i, j, l, bit, pl;
	unsigned long long x, t, r[10];

	if(f<=0||f>5)
		return len;

	if(len<=16)
		pl=0;
	else if(len<=24)
		pl=1;
	else if(len<=32)
		pl=2;
	else
		return len;

	memset(r, 0, sizeof(r));

    //reduce the polynomials
    bit=0;
    for(i=0;i<len;i++)
	{
        x=v[i];

        //special parity
        if(pl==2||f==5)
		{
            t=x;
            t^=t>>4;
            t^=t>>2;
            t^=t>>1;

            if(pl==2)
                r[0]^=(t&1)<<(i>>1);
            else
                r[0]^=(t&1)<<i;
            j=1;
        } else
            j=0;

        // cyclic polynomial case
        for(;j<2*f;j++)
            r[j]^=x<<(bit%xef_reg[f-1][pl][j]);
        bit+=8;
    }

    //pack the result (or rather, XOR over the original)
    for(i=0;i<2*f;i++)
	{
        l=xef_reg[f-1][pl][i];		//len
        x=r[i];
        x^=x>>l;
        for(j=0;j<l;j++)
		{
            v[bit>>3]^=((x>>j)&1)<<(bit&7);
            bit++;
        }
    }
    return bit;
}
unsigned	xef_fixerr(void *block, unsigned len, unsigned f)
{
    unsigned char *v=(unsigned char*)block;
    unsigned i, j, l, bit;
    unsigned long long r[10];
    unsigned pl, th;

    if(f<=0||f>5)
        return len;

	if(len<=16)
		pl=0;
	else if(len<=24)
		pl = 1;
	else if(len<=32)
		pl = 2;
	else
		return len;

	//unpack the registers
	memset(r, 0, sizeof(r));
	bit=len<<3;
	for(i=0;i<2*f;++i)
	{
		l=xef_reg[f-1][pl][i];		//len
		for(j=0;j<l;++j)
		{
			r[i]^=((unsigned long long)((v[bit>>3]>>(bit&7))&1))<<j;
			++bit;
		}
	}

    //fix errors
    for(i=0;i<(len<<3);++i)
	{
		th=7-f;
		if (pl==2)
			th+=(unsigned)(r[0]>>(i>>4))&1, j=1;
		else if (f==5)
			th+=(unsigned)(r[0]>>(i>>3))&1, j=1;
		else
			j=0;
        for(;j<2*f;++j)
            th+=(unsigned)(r[j]>>(i% xef_reg[f-1][pl][j]))&1;
        //if th > f
        v[i>>3]=(unsigned char)(v[i>>3]^((th>>3)<<(i&7)));
    }
    return bit;//return the true length
}
void		r5_set_parameters(R5_Parameters &p, const unsigned char tau, const unsigned char kappa_bytes, const unsigned short d, const unsigned short n, const unsigned short h,
	const unsigned char q_bits, const unsigned char p_bits, const unsigned char t_bits, const unsigned char b_bits,
	const unsigned short n_bar, const unsigned short m_bar, const unsigned char f, const unsigned char xe)
{
	p.kappa_bytes=kappa_bytes;
	p.d=d;
	p.n=n;
	p.h=h;
	p.q_bits=q_bits;
	p.p_bits=p_bits;
	p.t_bits=t_bits;
	p.b_bits=b_bits;
	p.n_bar=n_bar;
	p.m_bar=m_bar;
	p.f=f;
	p.xe=xe;

	//Derived parameters
	p.kappa=(unsigned short)(8*kappa_bytes);
	p.k=(unsigned short)(n?d/n:0);//Avoid arithmetic exception if n = 0
	p.mu=(unsigned short)(b_bits?((p.kappa+p.xe+b_bits-1)/b_bits):0); //Avoid arithmetic exception if B = 0
	p.q=(unsigned)(1U<<q_bits);
	p.p=(unsigned short)(1U<<p_bits);

	//Message sizes
	p.pk_size=(unsigned)(kappa_bytes+(d*n_bar*p_bits+7)/8);
	p.ct_size=(unsigned short)((d*m_bar*p_bits+7)/8 + (p.mu*t_bits+7)/8);

	//Rounding constants
	p.z_bits=(unsigned short)(p.q_bits-p.p_bits+p.t_bits);
	if(p.z_bits<p.p_bits)
		p.z_bits=p.p_bits;
	p.h1=(unsigned short)((unsigned short)1<<(p.q_bits-p.p_bits-1));
	p.h2=(unsigned short)(1<<(p.q_bits-p.z_bits-1));
	p.h3=(unsigned short)((unsigned short)(1<<(p.p_bits-p.t_bits-1)) + (unsigned short)(1<<(p.p_bits-p.b_bits-1)) - (unsigned short)(1<<(p.q_bits-p.z_bits-1)));

	//tau
	p.tau=p.k==1?0:tau;
	//set_parameter_tau(params, tau);
}
void		r5_cpa_pke_keygen(unsigned char *pk, unsigned char *sk, R5_Parameters const &p)
{
#ifdef R5_USE_IDX
	unsigned a_size=p.k*p.k*p.n, s_size=p.k*p.n_bar*p.h, b_size=p.k*p.n_bar*p.n;
#else
	unsigned a_size=p.k*p.k*p.n, s_size=p.k*p.n_bar*p.n, b_size=s_size;
#endif

	unsigned char *sigma=(unsigned char*)malloc(p.kappa_bytes);
	unsigned short *A=(unsigned short*)malloc(a_size*sizeof(unsigned short));
	short *S=(short*)malloc(s_size*sizeof(short));
	short *S_T=(short*)malloc(s_size*sizeof(short));
	unsigned short *B=(unsigned short*)malloc(b_size*sizeof(unsigned short));

	//memset(sigma, 0, p.kappa_bytes);
	//memset(A, 0, a_size*sizeof(unsigned short));
	//memset(S, 0, s_size*sizeof(short));
	//memset(S_T, 0, s_size*sizeof(short));
	//memset(B, 0, b_size*sizeof(unsigned short));
#ifdef PROFILER
	std::cout<<"KeyGen()\n";
	long long t1=__rdtsc();
#endif
	generate_uniform(p.kappa_bytes, sigma);
#ifdef PROFILER
	std::cout<<"generate_uniform: "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	r5_create_A(A, sigma, p);
#ifdef PROFILER
	std::cout<<"create_A:         "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	generate_uniform(p.kappa_bytes, sk);
#ifdef PROFILER
	std::cout<<"generate_uniform: "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
	r5_create_S_T(S_T, sk, p);
#ifdef PROFILER
	std::cout<<"create_S_T:       "<<__rdtsc()-t1<<endl;
	t1=__rdtsc();
#endif
#ifdef R5_USE_IDX
	r5_transpose_matrix((unsigned short*)S, (unsigned short*)S_T, p.n_bar, p.k, p.h);
#else
	r5_transpose_matrix((unsigned short*)S, (unsigned short*)S_T, p.n_bar, p.k, p.n);
#endif
#ifdef PROFILER
	std::cout<<"transpose_matrix: "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	r5_mult_matrix(B, (short*)A, p.k, S, p.k, p.n_bar, p.n, p.h, p.q, 0);//B = A S
#ifdef PROFILER
	std::cout<<"B = A S:          "<<__rdtsc()-t1<<endl;
#endif
#ifdef _DEBUG
	std::cout<<"sigma:\t", print_buffer(sigma, p.kappa_bytes);
	std::cout<<"A:";	for(int i=0;i<p.k*p.k;++i)		print_element((short*)A+i*p.n, p.n, p.q);
	std::cout<<"S_T:";	for(int i=0;i<p.n_bar*p.k;++i)	print_element(S_T+i*p.h, p.h, p.q);
	std::cout<<"B:";	for(int i=0;i<p.k*p.n_bar;++i)	print_element((short*)B+i*p.n, p.n, p.q);
#endif
	memcpy(pk, sigma, p.kappa_bytes);
	
#ifdef PROFILER
	t1=__rdtsc();
#endif
//	r5_pack_q_p(pk+p.kappa_bytes, B, p.h1, p);

	r5_round_matrix(B, p.k*p.n_bar, p.n, p.q_bits, p.p_bits, p.h1);
	r5_pack(pk+p.kappa_bytes, B, b_size, p.p_bits);
#ifdef PROFILER
	std::cout<<"pack_q_p:         "<<__rdtsc()-t1<<endl;
#endif

//#ifdef _DEBUG
//	std::cout<<"pk:\t", print_buffer(pk, p.pk_size);
//#endif
	free(sigma), free(A), free(S), free(S_T), free(B);
}
void		r5_cpa_pke_encrypt(unsigned char *ct, const unsigned char *pk, const unsigned char *m, const unsigned char *rho, R5_Parameters const &p)
{
	//Length of matrices, vectors, bit strings
    unsigned len_a=p.k*p.k*p.n;
#ifdef R5_USE_IDX
	unsigned len_r=p.k*p.m_bar*p.h;
#else
	unsigned len_r=p.k*p.m_bar*p.n;
#endif
    unsigned len_u=p.k*p.m_bar*p.n;
	unsigned len_b=p.k*p.n_bar*p.n;
	unsigned len_x=p.n_bar*p.m_bar*p.n;
	unsigned len_m1=(p.mu*p.b_bits+7)>>3;
	//Seeds
    unsigned char *sigma=(unsigned char*)malloc(p.kappa_bytes);
	//Matrices, vectors, bit strings
    unsigned short *A=(unsigned short*)malloc(len_a*sizeof(unsigned short));
    unsigned short *A_T=(unsigned short*)malloc(len_a*sizeof(unsigned short));
    short *R=(short*)malloc(len_r*sizeof(short));
    short *R_T=(short*)malloc(len_r*sizeof(short));
    unsigned short *U=(unsigned short*)malloc(len_u*sizeof(unsigned short));
    unsigned short *B=(unsigned short*)malloc(len_b*sizeof(unsigned short));
    unsigned short *B_T=(unsigned short*)malloc(len_b*sizeof(unsigned short));
    unsigned short *X=(unsigned short*)malloc(len_x*sizeof(unsigned short));
    unsigned short *v=(unsigned short*)malloc(p.mu*sizeof(unsigned short));
    unsigned char *m1=(unsigned char*)malloc(len_m1*sizeof(unsigned char));

	//memset(sigma, 0, p.kappa_bytes);
	//memset(A, 0, len_a*sizeof(unsigned short));
	//memset(A_T, 0, len_a*sizeof(unsigned short));
	//memset(R, 0, len_r*sizeof(short));
	//memset(R_T, 0, len_r*sizeof(short));
	//memset(U, 0, len_u*sizeof(unsigned short));
	//memset(B, 0, len_b*sizeof(unsigned short));
	//memset(B_T, 0, len_b*sizeof(unsigned short));
	//memset(X, 0, len_x*sizeof(unsigned short));
	//memset(v, 0, p.mu*sizeof(unsigned short));
	//memset(m1, 0, len_m1*sizeof(unsigned char));

	memcpy(sigma, pk, p.kappa_bytes);
	
#ifdef PROFILER
	std::cout<<"Encrypt()\n";
	long long t1=__rdtsc();
#endif
//	r5_unpack_p(B, pk+p.kappa_bytes, p);
	r5_unpack(B, pk+p.kappa_bytes, len_b, p.p_bits);
#ifdef PROFILER
	std::cout<<"unpack_p:         "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	r5_create_A(A, sigma, p);
#ifdef PROFILER
	std::cout<<"create_A:         "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	r5_create_S_T(R_T, rho, p);
#ifdef PROFILER
	std::cout<<"create_S_T:       "<<__rdtsc()-t1<<endl;

	t1=__rdtsc();
#endif
	r5_transpose_matrix(A_T, A, p.k, p.k, p.n);
#ifdef PROFILER
	std::cout<<"transpose_matrix: "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
#ifdef R5_USE_IDX
	r5_transpose_matrix((unsigned short*)R, (unsigned short*)R_T, p.k, p.k, p.h);
#else
	r5_transpose_matrix((unsigned short*)R, (unsigned short*)R_T, p.k, p.k, p.n);
#endif
#ifdef PROFILER
	std::cout<<"transpose_matrix: "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	r5_mult_matrix(U, (short*)A_T, p.k, R, p.k, p.m_bar, p.n, p.h, p.q, 0);//U = AT R
#ifdef PROFILER
	std::cout<<"U = AT R:         "<<__rdtsc()-t1<<endl;
#endif
#ifdef _DEBUG
	std::cout<<"m:\t", print_buffer(m, p.kappa_bytes);
	//std::cout<<"rho:\t", print_buffer(rho, p.kappa_bytes);
	//std::cout<<"sigma:\t", print_buffer(sigma, p.kappa_bytes);
	//std::cout<<"A:\t";	for(int i=0;i<p.k*p.k;++i)		print_element((short*)	A+i*p.n, p.n, p.q);
	//std::cout<<"B:\t";	for(int i=0;i<p.k*p.n_bar;++i)	print_element((short*)	B+i*p.n, p.n, p.q);
	//std::cout<<"R_T:\t";for(int i=0;i<p.k*p.m_bar;++i)	print_element(			R_T+i*p.h, p.h, p.q);
	std::cout<<"U:\t";	for(int i=0;i<p.k*p.m_bar;++i)	print_element((short*)	U+i*p.n, p.n, p.q);
#endif
#ifdef PROFILER
	t1=__rdtsc();
#endif
	r5_round_matrix(U, p.k*p.m_bar, p.n, p.q_bits, p.p_bits, p.h2);
#ifdef PROFILER
	std::cout<<"round_matrix:     "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	r5_transpose_matrix(B_T, B, p.k, p.n_bar, p.n);
#ifdef PROFILER
	std::cout<<"transpose_matrix: "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	r5_mult_matrix(X, (short*)B_T, p.n_bar, R, p.k, p.m_bar, p.n, p.h, p.p, p.xe!=0||p.f!=0);//X = BT R
#ifdef PROFILER
	std::cout<<"X = BT R:         "<<__rdtsc()-t1<<endl;
#endif
//#ifdef _DEBUG
//	std::cout<<"X = BT R\t", print_element((short*)X, p.mu, p.q);
//#endif
#ifdef PROFILER
	t1=__rdtsc();
#endif
	r5_round_matrix(X, p.mu, 1, p.p_bits, p.t_bits, p.h2);
#ifdef PROFILER
	std::cout<<"round_matrix:     "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	memcpy(m1, m, p.kappa_bytes);
	memset(m1+p.kappa_bytes, 0, len_m1-p.kappa_bytes);
	if(p.xe!=0)
		xef_compute(m1, p.kappa_bytes, p.f);
	{//add_msg
		int scale_shift=p.t_bits-p.b_bits;
		unsigned short val;
		unsigned bits_done=0, idx, bit_idx;
		memcpy(v, X, p.mu*sizeof(unsigned short));
		for(unsigned i=0;i<p.mu;++i)
		{
			idx=bits_done>>3, bit_idx=bits_done&7;
			val=m1[idx]>>bit_idx;
			if(bit_idx+p.b_bits>8)
				val|=m1[idx+1]<<(8-bit_idx);//Get spill over from next message byte
			v[i]=(v[i]+(val<<scale_shift))&((1<<p.t_bits)-1);
			bits_done+=p.b_bits;
		}
	}
#ifdef PROFILER
	std::cout<<"add_msg:          "<<__rdtsc()-t1<<endl;
#endif
#ifdef _DEBUG
	std::cout<<"U:\t"; for(int i=0;i<p.k*p.m_bar;++i)print_element((short*)U+i*p.n, p.n, p.q);
	std::cout<<"v:\t", print_element((short*)v, p.mu, p.q);
#endif
	
#ifdef PROFILER
	t1=__rdtsc();
#endif
//	r5_pack_q_p(ct, U, p.h2, p);
//#ifdef PROFILER
//	std::cout<<"pack U:           "<<__rdtsc()-t1<<endl;
//	t1=__rdtsc();
//#endif
//	{
//		const unsigned ndp_size=(p.n*p.p_bits+7)>>3, mut_size=(p.mu*p.t_bits+7)>>3;
//		memset(ct+ndp_size, 0, mut_size);
//		for(unsigned i=0, j=ndp_size<<3;i<p.mu;++i)//compute, pack v
//		{
//			unsigned short t=(X[i]+p.h2)>>(p.p_bits-p.t_bits);//compress p->t
//			unsigned short tm=m1[(i*p.b_bits)>>3]>>((i*p.b_bits)&7);//add message
//			if((p.b_bits&7)!=0)
//				if((i*p.b_bits&7)+p.b_bits> 8)
//					tm|=m1[((i*p.b_bits)>>3)+1]<<(8-((i*p.b_bits)&7));//Get spill over from next message byte
//			t=(t+((tm&((1<<p.b_bits)-1))<<(p.t_bits-p.b_bits)))&((1<<p.t_bits)-1);
//
//			ct[j>>3]|=t<<(j&7);//pack t bits
//			if((j&7)+p.t_bits>8)
//				ct[(j>>3)+1]|=t>>(8-(j&7));
//			j+=p.t_bits;
//		}
//	}

	{//pack_ct
		unsigned idx=r5_pack(ct, U, len_u, p.p_bits);
		r5_pack(ct+idx, v, p.mu, p.t_bits);
	}
#ifdef PROFILER
//	std::cout<<"compute, pack v:  "<<__rdtsc()-t1<<endl;
//	std::cout<<"pack U, v:        "<<__rdtsc()-t1<<endl;
#endif

//#ifdef _DEBUG
	//std::cout<<"v:\t", print_element((short*)v, p.mu, p.q);
	//std::cout<<"m1:\t", print_buffer(m1, len_m1);
//#endif
	free(sigma), free(A), free(A_T), free(R), free(R_T), free(U), free(B), free(B_T), free(X), free(v), free(m1);
}
void		r5_cpa_pke_decrypt(unsigned char *m, const unsigned char *sk, const unsigned char *ct, R5_Parameters const &p)
{
	//Length of matrices, vectors, bit strings
#ifdef R5_USE_IDX
	unsigned len_s=p.k*p.n_bar*p.h;
#else
	unsigned len_s=p.k*p.n_bar*p.n;
#endif
	unsigned len_u=p.k*p.m_bar*p.n;
	unsigned len_x_prime=p.n_bar*p.m_bar*p.n;
	unsigned len_m1=(p.mu*p.b_bits+7)>>3;//Message plus error correction
	//Matrices, vectors, bit strings
	short *S_T=(short*)malloc(len_s*sizeof(short));
	unsigned short *U		=(unsigned short*)malloc(len_u*sizeof(unsigned short));
	unsigned short *v		=(unsigned short*)malloc(p.mu*sizeof(unsigned short));
	unsigned short *X_prime	=(unsigned short*)malloc(len_x_prime*sizeof(unsigned short));
	unsigned short *m2		=(unsigned short*)malloc(p.mu*sizeof(unsigned short));
	unsigned char *m1=(unsigned char*)calloc(len_m1, 1);
	
#ifdef PROFILER
	std::cout<<"Decrypt()\n";
	long long t1=__rdtsc();
#endif
    r5_create_S_T(S_T, sk, p);
#ifdef PROFILER
	std::cout<<"create_S_T:       "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
//	r5_unpack_p(U, ct, p);
//#ifdef PROFILER
//	std::cout<<"unpack U:         "<<__rdtsc()-t1<<endl;
//	t1=__rdtsc();
//#endif
//	{
//		const unsigned ndp_size=(p.n*p.p_bits+7)>>3, mut_size=(p.mu*p.t_bits+7)>>3;
//		for(unsigned i=0, j=ndp_size<<3;i<p.mu;++i)
//		{
//			unsigned short t=ct[j>>3]>>(j&7);//unpack t bits
//			if((j&7)+p.t_bits>8)
//				t|=ct[(j>>3)+1]<<(8-(j&7));
//			v[i]=t&((1<<p.t_bits)-1);
//			j+=p.t_bits;
//		}
//	}

	{//unpack_ct
		unsigned idx=r5_unpack(U, ct, len_u, p.p_bits);
		r5_unpack(v, ct+idx, p.mu, p.t_bits);
	}
#ifdef PROFILER
//	std::cout<<"unpack v:         "<<__rdtsc()-t1<<endl;
	std::cout<<"unpack U, v:      "<<__rdtsc()-t1<<endl;
#endif

//#ifdef _DEBUG
//	std::cout<<"S_T:"; for(int i=0;i<p.n_bar*p.k;++i)print_element(S_T+i*p.h, p.h, p.q);
//	std::cout<<"U:\t"; for(int i=0;i<p.k*p.m_bar;++i)print_element((short*)U+i*p.n, p.n, p.q);
//	std::cout<<"v:\t", print_element((short*)v, p.mu, p.q);
//#endif
#ifdef PROFILER
	t1=__rdtsc();
#endif
	{//decompress_matrix	Decompress v t -> p
		const unsigned short p_mask=(1<<p.p_bits)-1;//not b_bits!
		for(int i=0;i<p.mu*1;++i)
		{
			const unsigned short shift=p.p_bits-p.t_bits;
			v[i]=v[i]<<shift&p_mask;
		//	v[i]=v[i]<<shift&b_mask;
		}
	}
#ifdef PROFILER
	std::cout<<"decompress v      "<<__rdtsc()-t1<<endl;
	
	t1=__rdtsc();
#endif
	r5_mult_matrix_swapped(X_prime, S_T, p.n_bar, (short*)U, p.k, p.m_bar, p.n, p.h, p.p, p.xe!=0||p.f!=0);//X' = U S = ST U
#ifdef PROFILER
	std::cout<<"X' = ST U:        "<<__rdtsc()-t1<<endl;
#endif
//#ifdef _DEBUG
//	std::cout<<"v mod p:\t", print_element((short*)v, p.mu, p.q);
//	std::cout<<"X' = U S = ST U:\t", print_element((short*)X_prime, p.mu, p.q);
//#endif
#ifdef PROFILER
	t1=__rdtsc();
#endif
	{//diff_msg
		for(unsigned i=0;i<p.mu;++i)
			m2[i]=(v[i]-X_prime[i])&(p.p-1);
	}
#ifdef PROFILER
	std::cout<<"diff_msg:         "<<__rdtsc()-t1<<endl;
#endif
//#ifdef _DEBUG
	//std::cout<<"m2:\t", print_element((short*)m2, p.mu, p.p);
//#endif
	
#ifdef PROFILER
	t1=__rdtsc();
#endif
	r5_round_matrix(m2, p.mu, 1, p.p_bits, p.b_bits, p.h3);
#ifdef PROFILER
	std::cout<<"round_matrix:     "<<__rdtsc()-t1<<endl;
#endif
#ifdef _DEBUG
	std::cout<<"round(m2):\t", print_element((short*)m2, p.mu, p.p);
#endif
	
#ifdef PROFILER
	t1=__rdtsc();
#endif
	r5_pack(m1, m2, p.mu, p.b_bits);
#ifdef PROFILER
	std::cout<<"pack m:           "<<__rdtsc()-t1<<endl;
#endif
	if(p.xe!=0)
	{
		xef_compute(m1, p.kappa_bytes, p.f);
		xef_fixerr(m1, p.kappa_bytes, p.f);
	}

	memcpy(m, m1, p.kappa_bytes);
#ifdef _DEBUG
	std::cout<<"m:\t", print_buffer(m, p.kappa_bytes);
#endif
    free(S_T), free(U), free(v), free(X_prime), free(m2), free(m1);
}
void		r5_cca_kem_keygen(unsigned char *pk, unsigned char *sk, R5_Parameters const &p)
{
	r5_cpa_pke_keygen(pk, sk, p);

	unsigned char *y=(unsigned char*)malloc(p.kappa_bytes);
	generate_uniform(p.kappa_bytes, y);
	memcpy(sk+p.kappa_bytes, y, p.kappa_bytes);
	memcpy(sk+p.kappa_bytes+p.kappa_bytes, pk, p.pk_size);
	free(y);
}
void		r5_cca_kem_encapsulate(unsigned char *ct, unsigned char *k, const unsigned char *pk, R5_Parameters const &p)
{
	//Allocate space
	unsigned char *hash_input=(unsigned char*)malloc(p.kappa_bytes+p.pk_size);
	unsigned char *m=(unsigned char*)malloc(p.kappa_bytes);
	unsigned char *L_g_rho=(unsigned char*)malloc(3U*p.kappa_bytes);

	//Generate random m
	generate_uniform(p.kappa_bytes, m);

	//Determine l, g, and rho
	memcpy(hash_input, m, p.kappa_bytes);
	memcpy(hash_input+p.kappa_bytes, pk, p.pk_size);
	FIPS202_SHAKE128(hash_input, p.kappa_bytes+p.pk_size, L_g_rho, 3U*p.kappa_bytes);

//#ifdef _DEBUG
//	std::cout<<"m:\t", print_buffer(m, p.kappa_bytes);
//	std::cout<<"L:\t", print_buffer(L_g_rho, p.kappa_bytes);
//	std::cout<<"g:\t", print_buffer(L_g_rho+p.kappa_bytes, p.kappa_bytes);
//	std::cout<<"rho:\t", print_buffer(L_g_rho+2*p.kappa_bytes, p.kappa_bytes);
//#endif

	//Encrypt m: ct = (U,v)
	r5_cpa_pke_encrypt(ct, pk, m, L_g_rho+2*p.kappa_bytes, p);

	//Append g: ct = (U,v,g)
	memcpy(ct+p.ct_size, L_g_rho+p.kappa_bytes, p.kappa_bytes);

	//k = H(L, ct)
	hash_input=(unsigned char*)realloc(hash_input, p.kappa_bytes+p.ct_size+p.kappa_bytes);
	memcpy(hash_input, L_g_rho, p.kappa_bytes);
	memcpy(hash_input+p.kappa_bytes, ct, p.ct_size+p.kappa_bytes);
	FIPS202_SHAKE128(hash_input, p.kappa_bytes+p.ct_size+p.kappa_bytes, k, p.kappa_bytes);

	free(hash_input), free(L_g_rho), free(m);
}
void		r5_cca_kem_decapsulate(unsigned char *k, const unsigned char *ct, const unsigned char *sk, R5_Parameters const &p)
{
	//Allocate space
	unsigned char *hash_input=(unsigned char*)malloc(p.kappa_bytes+p.pk_size);
	unsigned char *m_prime=(unsigned char*)malloc(p.kappa_bytes);
	unsigned char *L_g_rho_prime=(unsigned char*)malloc(3U*p.kappa_bytes);
	unsigned char *ct_prime=(unsigned char*)malloc(p.ct_size+p.kappa_bytes);
	const unsigned char *y=sk+p.kappa_bytes;//y is located after the sk
	const unsigned char *pk=y+p.kappa_bytes;//pk is located after y

	//Decrypt m'
	r5_cpa_pke_decrypt(m_prime, sk, ct, p);

	//Determine l, g, and rho
	memcpy(hash_input, m_prime, p.kappa_bytes);
	memcpy(hash_input+p.kappa_bytes, pk, p.pk_size);
	FIPS202_SHAKE128(hash_input, p.kappa_bytes+p.pk_size, L_g_rho_prime, 3U*p.kappa_bytes);

//#ifdef _DEBUG
//	std::cout<<"m_prime:\t", print_buffer(m_prime, p.kappa_bytes);
//	std::cout<<"L_prime:\t", print_buffer(L_g_rho_prime, p.kappa_bytes);
//	std::cout<<"g_prime:\t", print_buffer(L_g_rho_prime+p.kappa_bytes, p.kappa_bytes);
//	std::cout<<"rho_prime:\t", print_buffer(L_g_rho_prime+2*p.kappa_bytes, p.kappa_bytes);
//#endif

	//Encrypt m: ct' = (U',v')
	r5_cpa_pke_encrypt(ct_prime, pk, m_prime, L_g_rho_prime+2*p.kappa_bytes, p);
	//Append g': ct' = (U',v',g')
	memcpy(ct_prime+p.ct_size, L_g_rho_prime+p.kappa_bytes, p.kappa_bytes);

	//k = H(L', ct') or k = H(y, ct') depending on fail status
	hash_input=(unsigned char*)realloc(hash_input, p.kappa_bytes+p.ct_size+p.kappa_bytes);

	int error=0;
	for(int i=0;i<p.ct_size+p.kappa_bytes;++i)
		error|=ct[i]^ct_prime[i];
//	unsigned char fail=verify(ct, ct_prime, p.ct_size+p.kappa_bytes);

	memcpy(hash_input, L_g_rho_prime, p.kappa_bytes);
	memcpy(hash_input+p.kappa_bytes, ct_prime, p.ct_size+p.kappa_bytes);

	for(int i=0, flag=-(error|-error)>>7;i<p.kappa_bytes;++i)//constant-time conditional memcpy
		hash_input[i]^=flag&(hash_input[i]^y[i]);
//	conditional_constant_time_memcpy(hash_input, y, p.kappa_bytes, fail); //Overwrite L' with y in case of failure

	FIPS202_SHAKE128(hash_input, p.kappa_bytes+p.ct_size+p.kappa_bytes, k, p.kappa_bytes);
//	hash(k, p.kappa_bytes, hash_input, p.kappa_bytes+p.ct_size+p.kappa_bytes, p.kappa_bytes);

	free(hash_input), free(m_prime), free(L_g_rho_prime), free(ct_prime);
}
int			main()
{
	QueryPerformanceFrequency(&li);
	freq=li.QuadPart;
#if PROCESSOR_ARCH>=AVX2
	std::cout<<"AVX2";
#elif PROCESSOR_ARCH>=SSE3
	std::cout<<"SSE3";
#elif PROCESSOR_ARCH>=IA_32
	std::cout<<"IA32";
#endif
	int success=CryptAcquireContextA(&hProv, 0, 0, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT);
	if(use_rand=!success)
		std::cout<<"\trand()\n";
	//	std::cout<<"using stdlib/rand()\n";
	else
		std::cout<<"\tCryptGenRandom()\n";
	//	std::cout<<"using WinCrypt/CryptGenRandom()\n";
/*	{
		int cpui[4]={0};
		__cpuid(cpui, 0);//function IDs
		int nIds=cpui[0];
	//	std::vector<int[4]> data(nIds);
		int d_size=nIds<<2;//4 ints
		int *data=(int*)malloc(d_size*sizeof(int));
		for(int k=0;k<nIds;++k)
		//	__cpuidex(data[k], k, 0);
			__cpuidex(data+(k<<2), k, 0);
		char vendor[32]={0};
		((int*)vendor)[0]=data[1], ((int*)vendor)[1]=data[3], ((int*)vendor)[2]=data[2];
		bool isIntel=!memcmp(vendor, "GenuineIntel", 12), isAMD=!memcmp(vendor, "AuthenticAMD", 12);
		std::cout<<vendor<<endl;
		int f1_ecx, f1_edx, f7_ebx, f7_ecx, f81_ecx, f81_edx;
		if(nIds>=1)
			f1_ecx=data[(1<<2)+2], f1_edx=data[(1<<2)+3];
		if(nIds>=7)
			f7_ebx=data[(7<<2)+1], f7_ecx=data[(7<<2)+2];
		__cpuid(cpui, 0x80000000);//extended IDs
		unsigned nExIds=cpui[0];
		char brand[64]={0};
		struct CID
		{
			int i[4];
			CID(int *j){memcpy(i, j, 16);}
		};
		std::vector<CID> exdata;
		for(unsigned k=0x80000000;k<nExIds;++k)
		{
			__cpuidex(cpui, k, 0);
			exdata.push_back(CID(cpui));
		//	exdata.push_back(_mm_set_epi32(cpui[3], cpui[2], cpui[1], cpui[0]));
		//	exdata.push_back(*(__m128i*)cpui);
		}
		if(nExIds>=0x80000001u)
			f81_ecx=exdata[1].i[2], f81_edx=exdata[1].i[3];
		//	f81_ecx=exdata[1].m128i_i32[2], f81_edx=exdata[1].m128i_i32[3];
		if(nExIds>=0x80000004u)
		{
			memcpy(brand, &exdata[2], sizeof(int[4]));
			memcpy(brand+16, &exdata[3], sizeof(int[4]));
			memcpy(brand+32, &exdata[4], sizeof(int[4]));
		}
		std::cout<<brand<<endl;
		//auto getbit=[](int x, int i){return x>>i&1;};
#define GETBIT(x, i)	((x)>>(i)&1)
		bool sse3=GETBIT(f1_ecx, 0), pclmulqdq=GETBIT(f1_ecx, 1), monitor=GETBIT(f1_ecx, 3), sse3s=GETBIT(f1_ecx, 9), fma=GETBIT(f1_ecx, 12), cmpxchg16b=GETBIT(f1_ecx, 13), sse41=GETBIT(f1_ecx, 19), sse42=GETBIT(f1_ecx, 20), movbe=GETBIT(f1_ecx, 22), popcnt=GETBIT(f1_ecx, 23), aes=GETBIT(f1_ecx, 25), xsave=GETBIT(f1_ecx, 26), osxsave=GETBIT(f1_ecx, 27), avx=GETBIT(f1_ecx, 28), f16c=GETBIT(f1_ecx, 29), rdrand=GETBIT(f1_ecx, 30),
			msr=GETBIT(f1_edx, 5), cx8=GETBIT(f1_edx, 8), sep=GETBIT(f1_edx, 11), cmov=GETBIT(f1_edx, 15), clfsh=GETBIT(f1_edx, 19), mmx=GETBIT(f1_edx, 23), fxsr=GETBIT(f1_edx, 24), sse=GETBIT(f1_edx, 25), sse2=GETBIT(f1_edx, 26),
			fsgsbase=GETBIT(f7_ebx, 0), bmi1=GETBIT(f7_ebx, 3), hle=isIntel&&GETBIT(f7_ebx, 4), avx2=GETBIT(f7_ebx, 5), bmi2=GETBIT(f7_ebx, 8), erms=GETBIT(f7_ebx, 9), invpcid=GETBIT(f7_ebx, 10), rtm=isIntel&&GETBIT(f7_ebx, 11), avx512f=GETBIT(f7_ebx, 16), rdseed=GETBIT(f7_ebx, 18), adx=GETBIT(f7_ebx, 19), avx512pf=GETBIT(f7_ebx, 26), avx512er=GETBIT(f7_ebx, 27), avx512cd=GETBIT(f7_ebx, 28), sha=GETBIT(f7_ebx, 29),
			prefetchwt1=GETBIT(f7_ecx, 0),
			lahf=GETBIT(f81_ecx, 0), lzcnt=isIntel&&GETBIT(f81_ecx, 5), abm=isAMD&&GETBIT(f81_ecx, 5), sse4a=isAMD&&GETBIT(f81_ecx, 6), xop=isAMD&&GETBIT(f81_ecx, 11), tbm=isAMD&&GETBIT(f81_ecx, 21),
			syscall=isIntel&&GETBIT(f81_edx, 11), mmxext=isAMD&&GETBIT(f81_edx, 22), rdtscp=isIntel&&GETBIT(f81_edx, 27), _3dnowext=isAMD&&GETBIT(f81_edx, 30), _3dnow=isAMD&&GETBIT(f81_edx, 31);
		if(mmx)std::cout<<" MMX";
		if(mmxext)std::cout<<" MMXEXT";//AMD
		if(_3dnow)std::cout<<" 3DNOW";//AMD
		if(_3dnowext)std::cout<<" 3DNOWEXT";//AMD
		if(sse)std::cout<<" SSE";
		if(sse2)std::cout<<" SSE2";
		if(sse3)std::cout<<" SSE3";
		if(sse3s)std::cout<<" SSE3S";
		if(sse41)std::cout<<" SSE4.1";
		if(sse42)std::cout<<" SSE4.2";
		if(sse4a)std::cout<<" SSE4a";//AMD 2007
		if(xop)std::cout<<" XOP";//AMD 2009
		if(fma)std::cout<<" FMA";//AMD 2011
		if(avx)std::cout<<" AVX";
		if(avx2)std::cout<<" AVX2";
		if(avx512cd)std::cout<<" AVX512CD";
		if(avx512er)std::cout<<" AVX512ER";
		if(avx512f)std::cout<<" AVX512F";
		if(avx512pf)std::cout<<" AVX512PF";
		std::cout<<endl;

		if(abm)std::cout<<" ABM";
		if(adx)std::cout<<" ADX";
		if(aes)std::cout<<" AES";
		if(bmi1)std::cout<<" BMI1";
		if(bmi2)std::cout<<" BMI2";
		if(clfsh)std::cout<<" CLFSH";
		if(cmpxchg16b)std::cout<<" CMPXCHG16B";
		if(cx8)std::cout<<" CX8";
		if(erms)std::cout<<" ERMS";
		if(f16c)std::cout<<" F16C";
		if(fsgsbase)std::cout<<" FSGSBASE";
		if(fxsr)std::cout<<" FXSR";
		if(hle)std::cout<<" HLE";
		if(invpcid)std::cout<<" INVPCID";
		if(lahf)std::cout<<" LAHF";
		if(lzcnt)std::cout<<" LZCNT";
		if(monitor)std::cout<<" MONITOR";
		if(movbe)std::cout<<" MOVBE";
		if(msr)std::cout<<" MSR";
		if(cmpxchg16b)std::cout<<" CMPXCHG16B";
		if(osxsave)std::cout<<" OSXSAVE";
		if(pclmulqdq)std::cout<<" PCLMULQDQ";
		if(popcnt)std::cout<<" POPCNT";
		if(prefetchwt1)std::cout<<" PREFETCHWT1";
		if(cmpxchg16b)std::cout<<" CMPXCHG16B";
		if(rdrand)std::cout<<" RDRAND";
		if(rdseed)std::cout<<" RDSEED";
		if(rdtscp)std::cout<<" RDTSCP";
		if(rtm)std::cout<<" RTM";
		if(sep)std::cout<<" SEP";
		if(sha)std::cout<<" SHA";
		if(syscall)std::cout<<" SYSCALL";
		if(tbm)std::cout<<" TBM";
		if(xsave)std::cout<<" XSAVE";
		std::cout<<endl;
		free(data);
	}//*/

	//NTT benchmark
#if 0
//	const short n=1024, q=12289, w=49, sqrt_w=7, q_1=-12287/*=53249*/, n_1=12277, beta_q=4091, beta_1=2304, barrett_k=27, barrett_m=10921;//10921.8
//	const short n=512, q=12289, w=2401, sqrt_w=49, q_1=-12287, n_1=12277, beta_q=4091, beta_1=2304, barrett_k=27, barrett_m=10921;
	//X	const short n=512, q=19457, w=25, sqrt_w=5, q_1=-19455, n_1=19419, beta_q=7165, beta_1=5776, barrett_k=26, barrett_m=3449;//q too large
	const short n=256, q=7681, w=3844, sqrt_w=62, q_1=-7679/*57857*/, n_1=7651, beta_q=4088, beta_1=900, barrett_k=21, barrett_m=273;
//	const short n=128, q=769, w=49, sqrt_w=7, q_1=-767, n_1=763, beta_q=171, beta_1=9;//r2
//		const short n=128, q=257, w=9, sqrt_w=3, q_1=-255, n_1=255, beta_q=1, beta_1=1;//r2
//	const short n=64, q=641, w=441, sqrt_w=21, q_1=15745, n_1=631, beta_q=154, beta_1=487, barrett_k=19; int barrett_m=568644;
	//X	const short n=64, q=257, w=81, sqrt_w=9, q_1=-255/*65281*/, beta_q=1, beta_1=1, n_1=253, barrett_k=15, barrett_m=127;//127.5
//	const short n=32, q=10177, w=9575, sqrt_w=173, q_1=-6079, n_1=9859, beta_q=4474, beta_1=944, battett_k=25, barrett_m=3297;
	//	const short n=16, q=353, w=36, sqrt_w=6, beta_q=231, beta_1=217, q_1=25249, barrett_k=16, barrett_m=185;//185.6	//17, 371	IA32 only:
	//	const short n=8, q=193, w=9, sqrt_w=3, beta_q=109, beta_1=85, q_1=-28863, n_1=169, barrett_k=16, barrett_m=339;//r2
	//	const short n=4, q=41, w=9, sqrt_w=3, beta_q=18, beta_1=16, q_1=-25575, n_1=31, barrett_k=16, barrett_m=1598;
	//X	const short n=4, q=17, w=4, sqrt_w=2, q_1=-3855, beta_q=1, n_1=13, barrett_k=8, barrett_m=15;
	bool anti_cyclic=true;
	std::cout<<"\nNTT benchmark\t\tZ_"<<q<<"[x]/(x^"<<n<<"+1)\n\n";
	//int n_4=n/4;
	//int inv=0;
	//bool found=extended_euclidean_algorithm(4474, 10177, inv);
	//bool found=extended_euclidean_algorithm(q, 65536, inv);//25249
	//bool found=extended_euclidean_algorithm(q, 65536, inv);//61681
	//extended_euclidean_algorithm(4091, q, inv);//65536^-1 mod q=2304
	//extended_euclidean_algorithm(q, 65536, inv);//12289^-1 mod 65536=53249=-12287
	//extended_euclidean_algorithm(20479, 32768, inv);//53249
	//if(!extended_euclidean_algorithm(n, q, n_1))
	//	std::cout<<n<<" has no inverse mod "<<q<<endl;
	//else
	{
		int logn=log_2(n);
		NTT_params p;
		number_transform_initialize(n, q, w, sqrt_w, anti_cyclic, p);
	//	number_transform_initialize(n, q, beta_q, n_1, q_1, logn, w, sqrt_w, p);
		const int align=sizeof(SIMD_type);
#ifdef NTT_MULTIPLICATION
		short *a1=(short*)_aligned_malloc(n*sizeof(short), align), *a2=(short*)_aligned_malloc(n*sizeof(short), align);
		short *a1_2=(short*)_aligned_malloc(n*sizeof(short), align), *a2_2=(short*)_aligned_malloc(n*sizeof(short), align), *a3=(short*)_aligned_malloc(n*sizeof(short), align);
		short *ntt_a1=(short*)_aligned_malloc(n*sizeof(short), align), *ntt_a2=(short*)_aligned_malloc(n*sizeof(short), align), *ntt_a3=(short*)_aligned_malloc(n*sizeof(short), align);
#else
		short *src1=(short*)_aligned_malloc(n*sizeof(short), align), *src2=(short*)_aligned_malloc(n*sizeof(short), align), *dst=(short*)_aligned_malloc(n*sizeof(short), align);
		//	short *src1=new short[n], *src2=new short[n], *dst=new short[n];
		//	short src1[n]={0}, src2[n]={0}, dst[n]={0};
#endif
		for(char c=0;(c&0xDF)!='X';)
		{
#ifdef NTT_MULTIPLICATION
			for(int k=0;k<n;++k)
			//	a2[k]=a1[k]=rand()%q;
			//	a2[k]=a1[k]=k+1;
				a2[k]=a1[k]=1;
#else
			for(int k=0;k<n;++k)
			//	src1[k]=rand()%q;
			//	src1[k]=k+1;
				src1[k]=1;
			memcpy(src2, src1, n*sizeof(short));
#endif

			long long t1, t_total=0;
			int k=0, n_window=1, k2=0, k2End=1;
			for(k=0, n_window=100;k<n_window;++k)
			{
			//	unsigned aux=0;
			//	t1=__rdtscp(&aux);
				t1=__rdtsc();
				//QueryPerformanceCounter(&li);
				//t1=li.QuadPart;
				for(k2=0, k2End=1000;k2<k2End;++k2)
				{
#ifdef NTT_MULTIPLICATION
					memcpy(a1_2, a1, n*sizeof(short));
					memcpy(a2_2, a2, n*sizeof(short));
					apply_NTT(a1_2, ntt_a1, p);
					apply_NTT(a2_2, ntt_a2, p);
					multiply_ntt(ntt_a3, ntt_a1, ntt_a2, p);
				//	multiply_ntt(ntt_a3, ntt_a1, ntt_a2, n, q, q_1, beta_q);
					apply_inverse_NTT(ntt_a3, a3, p);
#else
					apply_NTT(src2, dst, p);
					apply_inverse_NTT(dst, src2, p);
				//	apply_NTT_modifies_src(src2, dst, q, q_1, n, p, anti_cyclic);	//dst[0], dst[1], dst[2], dst[3];
				//	apply_inverse_NTT(dst, src2, q, q_1, n, p, anti_cyclic);		//src2[0], src2[1], src2[2], src2[3];
#endif
				}
			//	long long t2=__rdtscp(&aux);
				long long t2=__rdtsc();
				t_total+=t2-t1;
				std::cout<<(double(t2-t1)/(freq*k2End))<<" ms\n";

				//QueryPerformanceCounter(&li);
				//t_total+=li.QuadPart-t1;
				//std::cout<<(1000.*(li.QuadPart-t1)/freq)<<" ms\n";
			//	printf("%lf ms\n", 1000.*(li.QuadPart-t1)/freq);
			//	std::cout<<' '<<(1000.*(li.QuadPart-t1)/freq);
			}
		//	std::cout<<endl;
			int n_repetitions=n_window*k2End;
			std::cout<<"\tAverage: "<<(double(t_total)/(freq*n_repetitions))<<" ms";
			std::cout<<"\tTicks: "<<(double(t_total)/n_repetitions);
			//std::cout<<"\tAverage: "<<(1000.*t_total/freq/n_window)<<" ms";
			//std::cout<<"\tTicks: "<<(t_total/n_window);
		//	printf("\tAverage: %lf ms", 1000.*t_total/freq/n_window);
		//	std::cout<<"average: "<<(1000.*t_total/freq/n_window)<<" ms\n";

#ifdef NTT_MULTIPLICATION
			{
				std::cout<<"\nToom-Cook4 method (q=2^13):\n";
				t_total=0;
				k=0; int n_window2=1; k2=0, k2End=1;
				for(k=0, n_window2=10;k<n_window2;++k)
				{
					t1=__rdtsc();
					for(k2=0, k2End=1000;k2<k2End;++k2)
						multiply_toom_cook4_saber(a1, a2, ntt_a3, n, 13, anti_cyclic);
					long long t2=__rdtsc();
					t_total+=t2-t1;
					std::cout<<(double(t2-t1)/(freq*k2End))<<" ms\n";
				}
				n_repetitions=n_window2*k2End;
				std::cout<<"\tAverage: "<<(double(t_total)/(freq*n_repetitions))<<" ms";
				printf("\tTicks: %lf\n", (double(t_total)/n_repetitions));
			}
			{
				std::cout<<"\nKaratsuba method (q=2^13):\n";
				t_total=0;
				k=0; int n_window2=1; k2=0, k2End=1;
				for(k=0, n_window2=10;k<n_window2;++k)
				{
					t1=__rdtsc();
					for(k2=0, k2End=1000;k2<k2End;++k2)
						multiply_karatsuba(a1, a2, ntt_a3, n, 13, anti_cyclic);
					long long t2=__rdtsc();
					t_total+=t2-t1;
					std::cout<<(double(t2-t1)/(freq*k2End))<<" ms\n";
				}
				n_repetitions=n_window2*k2End;
				std::cout<<"\tAverage: "<<(double(t_total)/(freq*n_repetitions))<<" ms";
				printf("\tTicks: %lf\n", (double(t_total)/n_repetitions));
			}
			std::cout<<"\nNaive method:\t";
		//	t_total=0;
			t1=__rdtsc();
			multiply_polynomials(a1, a2, ntt_a3, n, q, barrett_k, barrett_m, anti_cyclic);
			long long t2=__rdtsc();
		//	t_total+=t2-t1;
			std::cout<<(double(t2-t1)/freq)<<" ms, "<<(t2-t1)<<" cycles\n";
#endif
			//std::cout<<"\nsrc1:", print_element(src1, n, q);
			//std::cout<<"\nsrc2:", print_element(src2, n, q);
			int error=0;
			for(int k=0;k<n;++k)
#ifdef NTT_MULTIPLICATION
				error+=unsigned short(a3[k]^ntt_a3[k]);
#else
				error+=unsigned short(src1[k]^src2[k]);
#endif
			std::cout<<"\terror="<<error<<endl;

		//	std::cout<<"\nntt a3:", print_element(a3, n, q);
		//	std::cout<<"\nnaive a3:", print_element(ntt_a3, n, q);
			std::cout<<endl;
			c=_getch();
		}
#ifdef NTT_MULTIPLICATION
		_aligned_free(a1), _aligned_free(a2), _aligned_free(a3), _aligned_free(ntt_a1), _aligned_free(ntt_a2), _aligned_free(ntt_a3);
#else
		_aligned_free(src1), _aligned_free(src2), _aligned_free(dst);
#endif
		number_transform_destroy(p);
	}
#endif
/*	{
		int n=256, logq=13, q=1<<logq;
		short *a=new short[n], *b=new short[n], *c=new short[n];
		for(int k=0;k<n;++k)
		//	a[k]=rand(), b[k]=rand();
			a[k]=b[k]=1;
		//	a[k]=b[k]=k==2;
		memset(c, 0, n*sizeof(short));
		std::cout<<"a:", print_element(a, n, q);
		std::cout<<"b:", print_element(b, n, q);
		multiply_toom_cook4_saber(a, b, c, n, logq, true);
		std::cout<<"c = a b:", print_element(c, n, q);
		multiply_polynomials_mod_powof2_add(a, b, c, n, logq, true);
		std::cout<<"c = a b:", print_element(c, n, q);
		_getch();
		delete[] a, b, c;
	}//*/
	//{
	//	__m128i M[8];
	//	for(int k=0;k<8;++k)
	//		for(int k2=0;k2<8;++k2)
	//			M[k].m128i_i16[k2]=(k<<3)+k2;
	//	for(int k=0;k<8;++k)
	//		print_register(M[k], 0x7FFF);
	//	transpose(M);
	//	std::cout<<endl;
	//	for(int k=0;k<8;++k)
	//		print_register(M[k], 0x7FFF);
	//	_getch();
	//}
/*	{
		AES::initiate();
		unsigned char seed[32]={0};
		drbg_start(seed);
		for(;;)
		{
			long long t_gen=-1;
			unsigned char x;
			for(int k=0;k<16;++k)
			{
				long long t1=__rdtsc();
				drbg_get(&x, 1);
				long long t=__rdtsc()-t1;
				if(t_gen==-1||t_gen>t)
					t_gen=t;
			//	t_gen+=__rdtsc()-t1;
				printf("%02x", (int)x);
			}
			std::cout<<'\t'<<t_gen;
		//	std::cout<<'\t'<<double(t_gen)/16;

			std::cout<<endl;
			if((_getch()&0xDF)=='X')
				break;
		}
		//unsigned char str1[8]={0}, str2[64]={0}, str3[4]={0}, str4[4]={0}, str5[1]={0};
		//drbg_get(str1, 8);
		//drbg_get(str2, 64);
		//drbg_get(str3, 4);
		//drbg_get(str4, 4);
		//drbg_get(str5, 1);
		//print_buffer(str1, 8);
		//print_buffer(str2, 64);
		//print_buffer(str3, 4);
		//print_buffer(str4, 4);
		//print_buffer(str5, 1);
		//_getch();
	}//*/
/*	{
		AES::initiate();
		const unsigned block_size=16;
		char message[block_size+1]={0};
		unsigned char seed[11*16]={0};
		printf("seed:\t%016llx%016llx\n", ((long long*)seed)[0], ((long long*)seed)[1]);
		AES::expand_key(seed);
		unsigned long long ctr=~0, ctr2=~0;
		for(;;)
		{
			(long long&)message=ctr, ((long long*)&message)[1]=ctr2;
			printf("%016llx%016llx\n", ctr2, ctr);
		//	print_buffer(message, m_size);
			AES::encrypt((unsigned char*)message);
			print_buffer(message, block_size);
			AES::decrypt((unsigned char*)message);
			print_buffer(message, block_size);
			//std::cout<<endl;
			//AES::encrypt2((unsigned char*)message);
			//print_buffer(message, block_size);
			
			int flag=~ctr==0;
			++ctr, ctr&=(long long)-!flag, ctr2+=flag;

			std::cout<<endl;
			if((_getch()&0xDF)=='X')
				break;
		}
	}//*/
/*	{
		int m_size=0;
		char message[]="12345678901234567890123456789012";
		const int h_size=32;//0->32 39330, 32->2048 440610
		unsigned char hash[h_size]={0};
		long long t_min=-1;
		for(;;)
		{
			long long t1=__rdtsc();
		//	shake128(hash, h_size, (unsigned char*)message, m_size);//34820
			FIPS202_SHAKE128((unsigned char*)message, m_size, hash, h_size);//38020	40350
		//	FIPS202_SHAKE256((unsigned char*)message, m_size, hash, h_size);//37900
			t1=__rdtsc()-t1;
			print_buffer(hash, h_size);
			std::cout<<"7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26\n";//SHAKE128("")
		//	std::cout<<"46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be\n";//SHAKE256("")
			
			if(t_min==-1||t_min>t1)
				t_min=t1;
			std::cout<<t_min<<'\t'<<t1<<endl;

			if((_getch()&0xDF)=='X')
				break;
		}
	}//*/
/*	{
		printf("\n\nCopy at&t assembly to remove comments, and press a key\n");
		_getch();
		OpenClipboard(0);
		char *clipboard=(char*)GetClipboardData(CF_OEMTEXT);	if(!clipboard				){CloseClipboard();_getch();return 0;}
		int size=strlen(clipboard);								if(size<=0||size>0x003FFFFF	){CloseClipboard();_getch();return 0;}
		std::string str(clipboard), str2;
		for(int k=0;k<size;)
		{
			if(k>=65900)//65947
				int LOL_1=0;
			for(;k<size&&(str[k]==' '||str[k]=='\t'||str[k]=='\n');++k);//skip whitespace and empty lines
			if(k>=size)break;
			if(str[k]=='#')//skip comment
			{
				for(;k<size&&str[k]!='\n';++k);
				continue;
			}
			//line
			int k2=k;
			for(;k2<size&&str[k2]!='\n';++k2);
			str2.append(str.begin()+k, str.begin()+k2+1);
			k=k2;
		}
		char *clipboard2=(char*)LocalAlloc(LMEM_FIXED, (str2.size()+1)*sizeof(char));
		memcpy(clipboard2, str2.c_str(), str2.size()*sizeof(char));
		OpenClipboard(0), EmptyClipboard(), SetClipboardData(CF_OEMTEXT, (void*)clipboard2), CloseClipboard();
		printf("Removed comments.\n");
		_getch();
		return 0;
	}//*/
/*	{
		int n=1024;
		unsigned short *a=(unsigned short*)malloc(n*sizeof(short)), *omega=(unsigned short*)malloc(n*sizeof(short));
	//	int *a=(int*)malloc(n*sizeof(int));
	//	double *b1=(double*)malloc(n*sizeof(double)), *b2=(double*)malloc(n*sizeof(double));
	//	memset(b1, 0, n*sizeof(double)), memset(b2, 0, n*sizeof(double));
		for(int k=0;k<n;++k)
			a[k]=k%2, omega[k]=1;
		//	a[k]=k%2, b1[k]=b2[k]=1;

		ntt(a, omega);
	//	ntt_double(a, b1, b2);

		std::cout<<"a:\n";
		print_element((short*)a, n);
		std::cout<<"\n\nomega:\n";
		print_element((short*)omega, n);
		free(a), free(omega);

	//	int n=1024;
	//	int *a=(int*)malloc(n*sizeof(int));
	//	double *b1=(double*)malloc(n*sizeof(double)), *b2=(double*)malloc(n*sizeof(double));
	////	memset(b1, 0, n*sizeof(double)), memset(b2, 0, n*sizeof(double));
	//	for(int k=0;k<n;++k)
	//		a[k]=k%2, b1[k]=b2[k]=1;

	//	ntt_double(a, b1, b2);

	//	std::cout<<"a:\n";
	//	for(int k=0;k<n;++k)
	//	{
	//		for(int k2=0;k2<16;++k2)
	//			printf(" %5d", a[k+k2]);
	//		std::cout<<endl;
	//	}
	//	std::cout<<"b1:\n";
	//	for(int k=0;k<n;++k)
	//	{
	//		for(int k2=0;k2<16;++k2)
	//			printf(" %5lf", b1[k+k2]);
	//		std::cout<<endl;
	//	}
	//	std::cout<<"b2:\n";
	//	for(int k=0;k<n;++k)
	//	{
	//		for(int k2=0;k2<16;++k2)
	//			printf(" %5lf", b2[k+k2]);
	//		std::cout<<endl;
	//	}
	//	free(a), free(b1), free(b2);
		_getch();
		return 0;
	}//*/
	;
	//Round5 2018
#if 1
	const unsigned r5_parameter_sets[][16]={
		//C_SK,  C_PK,   C_B, C_CT, kappa_bytes, d, n, h, q_bits, p_bits, t_bits, b_bits, n_bar, m_bar, f, xe
		{   16,   634,    16,   682, 16,	  618,  618, 104, 11,	 8,		 4,		 1,		 1,		 1, 0U, 0},//R5ND_1KEM_0c
		{   24,   909,    24,   981, 24,	  786,  786, 384, 13,	 9,		 4,		 1,		 1,		 1, 0U, 0},//R5ND_3KEM_0c
		{   32,  1178,    32,  1274, 32,	 1018, 1018, 428, 14,	 9,		 4,		 1,		 1,		 1, 0U, 0},//R5ND_5KEM_0c
		{  708,   676,   756,     0, 16,	  586,  586, 182, 13,	 9,		 4,		 1,		 1,		 1, 0U, 0},//R5ND_1PKE_0c
		{ 1031,   983,  1119,     0, 24,	  852,  852, 212, 12,	 9,		 5,		 1,		 1,		 1, 0U, 0},//R5ND_3PKE_0c
		{ 1413,  1349,  1525,     0, 32,	 1170, 1170, 222, 13,	 9,		 5,		 1,		 1,		 1, 0U, 0},//R5ND_5PKE_0c
		{   16,  5214,    16,  5236, 16,	  594,    1, 238, 13,	10,		 7,		 3,		 7,		 7, 0U, 0},//R5N1_1KEM_0c
		{   24,  8834,    24,  8866, 24,	  881,    1, 238, 13,	10,		 7,		 3,		 8,		 8, 0U, 0},//R5N1_3KEM_0c
		{   32, 14264,    32, 14288, 32,	 1186,    1, 712, 15,	12,		 7,		 4,		 8,		 8, 0U, 0},//R5N1_5KEM_0c
		{ 5772,  5740,  5804,     0, 16,	  636,    1, 114, 12,	 9,		 6,		 2,		 8,		 8, 0U, 0},//R5N1_1PKE_0c
		{ 9708,  9660,  9732,     0, 24,	  876,    1, 446, 15,	11,		 7,		 3,		 8,		 8, 0U, 0},//R5N1_3PKE_0c
		{14700, 14636, 14724,     0, 32,	 1217,    1, 462, 15,	12,		 9,		 4,		 8,		 8, 0U, 0}//R5N1_5PKE_0c
	};
	const char *r5_parameter_set_names[]={
		"R5ND_1KEM_0c", "R5ND_3KEM_0c", "R5ND_5KEM_0c", "R5ND_1PKE_0c", "R5ND_3PKE_0c", "R5ND_5PKE_0c",
		"R5N1_1KEM_0c", "R5N1_3KEM_0c", "R5N1_5KEM_0c", "R5N1_1PKE_0c", "R5N1_3PKE_0c", "R5N1_5PKE_0c"
	};
#ifdef XOF_USE_DRBG_AES128
	AES::initiate();
#endif
	const int tau=0, api_set_number=1;
	const unsigned *set=r5_parameter_sets[api_set_number];
	R5_Parameters p;
	r5_set_parameters(p, tau, set[4], set[5], set[6], set[7], set[8], set[9], set[10], set[11], set[12], set[13], set[14], set[15]);
//	r5_set_parameters(p, tau, 1, 10, 10, 2,		13, 9, 4, 1, 1, 1, 0, 0);//
	printf("Round5 %s\n", r5_parameter_set_names[api_set_number]);
	
	long long tg=-1, te=-1, td=-1;
	for(;;)
	{
	/*	//Round5 CCA KEM
		unsigned sk_size=p.kappa_bytes+p.kappa_bytes+p.pk_size;
		unsigned char *sk=(unsigned char*)malloc(sk_size);
		unsigned char *pk=(unsigned char*)malloc(p.pk_size);
		long long t1=__rdtsc();
		r5_cca_kem_keygen(pk, sk, p);
		long long t_gen=__rdtsc()-t1;

		unsigned char *k1=(unsigned char*)malloc(p.kappa_bytes);
		unsigned ct_size=p.ct_size+p.kappa_bytes;
		unsigned char *ct=(unsigned char*)malloc(ct_size);
		t1=__rdtsc();
		r5_cca_kem_encapsulate(ct, k1, pk, p);
		long long t_enc=__rdtsc()-t1;
	
		unsigned char *k2=(unsigned char*)malloc(p.kappa_bytes);
		t1=__rdtsc();
		r5_cca_kem_decapsulate(k2, ct, sk, p);
		long long t_dec=__rdtsc()-t1;

		std::cout<<"k1:\t", print_buffer(k1, p.kappa_bytes);
		std::cout<<"k2:\t", print_buffer(k2, p.kappa_bytes);

		free(sk), free(pk), free(ct), free(k1), free(k2);//*/


		//Round5 CPA PKC
		unsigned char *sk=(unsigned char*)malloc(p.kappa_bytes);
		unsigned char *pk=(unsigned char*)malloc(p.pk_size);
		long long t1=__rdtsc();
		r5_cpa_pke_keygen(pk, sk, p);
		long long t_gen=__rdtsc()-t1;

		const unsigned m_size=32+1;
		char message[]="12345678901234567890123456789012";
	//	char message[m_size]={0};
		unsigned char *rho=(unsigned char*)malloc(p.kappa_bytes);
		generate_uniform(p.kappa_bytes, rho);
		unsigned char *ct=(unsigned char*)malloc(p.ct_size);
		t1=__rdtsc();
		r5_cpa_pke_encrypt(ct, pk, (unsigned char*)message, rho, p);
		long long t_enc=__rdtsc()-t1;

		char message2[m_size]={0};
		t1=__rdtsc();
		r5_cpa_pke_decrypt((unsigned char*)message2, sk, ct, p);
		long long t_dec=__rdtsc()-t1;
	//	std::cout<<"message:\t"<<message<<endl;
	//	std::cout<<"decryption:\t"<<message2<<endl;
		print_buffer(message, p.kappa_bytes);//
		print_buffer(message2, p.kappa_bytes);//
	
		free(sk), free(pk), free(ct);//*/


		if(tg==-1||tg>t_gen)
			tg=t_gen;
		if(te==-1||te>t_enc)
			te=t_enc;
		if(td==-1||td>t_dec)
			td=t_dec;
		std::cout<<"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		std::cout<<"Minimum:           "<<double(tg)/freq<<"ms "<<tg<<", "<<double(te)/freq<<"ms "<<te<<", "<<double(td)/freq<<"ms "<<td<<",  "<<double(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		
		std::cout<<endl;
		if((_getch()&0xDF)=='X')
			break;
	}
#endif
	//Saber 2018
#if 0
//	const short n=256, logq=13, logp=10, logt=2,	mu=10, k=2;//FireSaber
	const short n=256, logq=13, logp=10, logt=3,	mu=8, k=3;//Saber			TODO: saber_binomial_mu
//	const short n=256, logq=13, logp=10, logt=5,	mu=6, k=4;//LightSaber
//	const short n=32, logq=13, logp=10, logt=3,	mu=8, k=3;
//	const short n=16, logq=13, logp=10, logt=3,	mu=8, k=3;
//	const short n=8, logq=13, logp=10, logt=3,	mu=8, k=3;

	const short q=1<<logq, p=1<<logp, t=1<<logt;
	bool anti_cyclic=true;
	std::cout<<"Saber PKC\t\tZ_"<<q<<"[x]/(x^"<<n<<"+1)\n\n";
#ifdef XOF_USE_DRBG_AES128
	AES::initiate();
#endif
//	double sg=0, se=0, sd=0;
	long long tg=-1, te=-1, td=-1;
	for(int k_loop=1;;++k_loop)
//	for(int k_loop=1;k_loop<1000;++k_loop)
//	for(;;)
	{
	/*	//Saber.CCA KEM		page 12
		Saber_private_key pr_k;
		Saber_public_key pu_k;
		long long t1=__rdtsc();
		saber_generate(pr_k, pu_k, n);
		long long t_gen=__rdtsc()-t1;

		Saber_ciphertext ct;
		const int size=32;
		unsigned char K[size];
		t1=__rdtsc();
		saber_cca_encapsulate(pu_k, K, ct, n);
		long long t_enc=__rdtsc()-t1;
	//	std::cout<<"\tEncrypted";//
			
		unsigned char K2[(n>>3)+1]={0};
		t1=__rdtsc();
		saber_cca_decapsulate(ct, K2, pr_k, pu_k, n);
		long long t_dec=__rdtsc()-t1;
		
		std::cout<<"Ka:\t", print_buffer(K, n>>3);
		std::cout<<"Kb:\t", print_buffer(K2, n>>3);
		bool error=false;
		for(int kx=0;kx<n>>3;++kx)
			if(K2[kx]!=K[kx])
			{
				error=true;
				break;
			}
		if(error)
			std::cout<<"\tError\a";
		delete[] ct.b_dash, ct.cm;//destroy ciphertext
		delete[] pu_k.seed_A, pu_k.b, pr_k.s;//destroy keys//*/

		//Saber.CPA PKC		pages 10,11
		Saber_private_key pr_k;
		Saber_public_key pu_k;
		long long t1=__rdtsc();
		saber_generate(pr_k, pu_k, n);
		long long t_gen=__rdtsc()-t1;

		const char *message="12345678901234567890123456789012";//256bit
		Saber_ciphertext ct;
		const int size=32;
		unsigned char r[size];
		generate_uniform(size, r);
		t1=__rdtsc();
		saber_encrypt(message, ct, pu_k, r, n);
		long long t_enc=__rdtsc()-t1;
	//	std::cout<<"\tEncrypted";//
			
		char message2[(n>>3)+1]={0};
		t1=__rdtsc();
		saber_decrypt(ct, message2, pr_k, pu_k, n);
		long long t_dec=__rdtsc()-t1;
		std::cout<<"Message:\t"<<message<<endl;
		std::cout<<"Decryption:\t"<<message2<<endl;
		bool error=false;
		for(int kx=0;kx<n>>3;++kx)
			if(message2[kx]!=message[kx])
			{
				error=true;
				break;
			}
		if(error)
		{
			std::cout<<"\tError\a";
		//	saber_decrypt_DEBUG(ct, message2, pr_k, pu_k, n);
			std::cout<<endl;
			print_buffer(message, n>>3);//
			print_buffer(message2, n>>3);//
		}
		delete[] ct.b_dash, ct.cm;//destroy ciphertext
		delete[] pu_k.seed_A, pu_k.b, pr_k.s;//destroy keys//*/

		std::cout<<endl;
		//sg+=t_gen, se+=t_enc, sd+=t_dec;
		//double tg=sg/k_loop, te=se/k_loop, td=sd/k_loop;
		//std::cout<<	"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		//std::cout<<	"Average:           "<<tg/freq<<"ms "<<tg<<", "<<te/freq<<"ms "<<te<<", "<<td/freq<<"ms "<<td<<",  "<<(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		if(tg==-1||tg>t_gen)
			tg=t_gen;
		if(te==-1||te>t_enc)
			te=t_enc;
		if(td==-1||td>t_dec)
			td=t_dec;
		//i7-6800K AVX2:	
		//i7-6800K SSE:		0.431765ms 1433603, 0.793038ms 2633151, 0.115286ms 382787,  1.34009ms 4449541
		//i7-6800K IA32:	0.451328ms 1498560, 0.509264ms 1690925, 0.071189ms 236371,  1.03178ms 3425856
		//U7700 SSE:	1.68884ms 2193610, 1.97055ms 2559520, 0.306986ms 398740,  3.96638ms 5151870		//soft 1.77448ms 2304850, 2.06069ms 2676600, 0.311475ms 404570,  4.14665ms 5386020
		//U7700 IA32:	2.02023ms 2624050, 2.4079ms 3127590, 0.394115ms 511910,  4.82225ms 6263550
		std::cout<<"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		std::cout<<"Minimum:           "<<double(tg)/freq<<"ms "<<tg<<", "<<double(te)/freq<<"ms "<<te<<", "<<double(td)/freq<<"ms "<<td<<",  "<<double(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		//std::cout<<"Generation:\t"<<double(t_gen)/freq<<"ms, "<<t_gen<<"cycles\n";//1.79387ms, 2330030c		1.81238ms, 2354070c
		//std::cout<<"Encryption:\t"<<double(t_enc)/freq<<"ms, "<<t_enc<<"cycles\n";//2.09025ms, 2714990c		2.09519ms, 2721410c
		//std::cout<<"Decryption:\t"<<double(t_dec)/freq<<"ms, "<<t_dec<<"cycles\n";//0.324301ms, 421230c		0.320328ms, 416070c
		//std::cout<<"Total:\t"<<double(t_gen+t_enc+t_dec)/freq<<"ms, "<<(t_gen+t_enc+t_dec)<<"cycles\n";//4.20841ms, 5466250c	4.22789ms, 5491550c
		//*/

	/*	//Saber.KE	page 6
		int size=32, A_size=k*k*n, vector_size=k*n;
		//A:
	//	std::cout<<"\tAlice:";
		unsigned char *seed_A=new unsigned char[size];
		generate_uniform(size, seed_A);
		short *A=new short[A_size];
		FIPS202_SHAKE128(seed_A, size, (unsigned char*)A, A_size*sizeof(short));
	//	for(int k2=0;k2<A_size;++k2)A[k2]=1;//
		std::cout<<"\nA:"; for(int k2=0, k2End=k*k;k2<k2End;++k2)print_element_hex_nnl(A+n*k2, n, q);//

		short *s=new short[vector_size];
		saber_generate_binomial_8(s, vector_size);
	//	for(int k2=0;k2<vector_size;++k2)s[k2]=1;//
		std::cout<<"\ns:"; for(int k2=0;k2<k;++k2)print_element_small(s+n*k2, n, q);//

		short *b=new short[vector_size];
		saber_calculate_b(b, A, s, n, k, logq, logp, anti_cyclic);
		std::cout<<"\nb = A s + 4 >> 3 in Rp:"; for(int k2=0;k2<k;++k2)print_element_hex_nnl(b+n*k2, n, q);//
		delete[] A;

		//B:	<- b, seed_A
	//	std::cout<<"\n\n\tBob:";
		short *s_dash=new short[vector_size];
		saber_generate_binomial_8(s_dash, vector_size);
	//	for(int k2=0;k2<vector_size;++k2)s_dash[k2]=1;//
		std::cout<<"\ns':"; for(int k2=0;k2<k;++k2)print_element_small(s_dash+n*k2, n, q);//

		A=new short[A_size];
		FIPS202_SHAKE128(seed_A, size, (unsigned char*)A, A_size*sizeof(short));
	//	for(int k2=0;k2<A_size;++k2)A[k2]=1;//
		std::cout<<"\nA:"; for(int k2=0, k2End=k*k;k2<k2End;++k2)print_element_hex_nnl(A+n*k2, n, q);//

		short *b_dash=new short[vector_size];
		saber_calculate_b_dash(b_dash, A, s_dash, n, k, logq, logp, anti_cyclic);
		std::cout<<"\nb' = AT s' + 4 >> 3 in Rp:"; for(int k2=0;k2<k;++k2)print_element_hex_nnl(b_dash+n*k2, n, q);//

		short *v_dash=new short[n];
		saber_calculate_v(v_dash, b, s_dash, n, k, logq, logp, anti_cyclic);
		std::cout<<"\nv' = bT s' in Rp:", print_element_hex_nnl(v_dash, n, q);//

		short *c=new short[n];
		saber_bits(c, v_dash, n, logp-1, logt);
		//for(int kx=0, sh=logp-1-logt, mask=(1<<logt)-1;kx<n;++kx)
		//	c[kx]=v[kx]>>sh&mask;
		std::cout<<"\nc = v' >> ep-et-1 in Rt:", print_element_hex_nnl(c, n, q);//

		short *K_dash=new short[n];
		saber_bits(K_dash, v_dash, n, logp, 1);
		std::cout<<"\nK' = v' >> ep-1 in R2:", print_element_hex_nnl(K_dash, n, q);//
	//	unsigned char *KB=pack_bits((short*)K_dash, n, 1, 1, 1);
		unsigned char *KB=new unsigned char[size];
		saber_kdf(KB, K_dash, n);
		//memset(KB, 0, size);
		//for(int kx=0;kx<size;++kx)
		//	for(int k2=0;k2<8;++k2)
		//		KB[kx]|=K_dash[(kx<<3)+k2]<<k2;
		delete[] A, s_dash, v_dash, K_dash;

		//A:	<- b', c
		std::cout<<"\n\n\tAlice:";
		short *v=new short[n];
		saber_calculate_v(v, b_dash, s, n, k, logq, logp, anti_cyclic);
		std::cout<<"\nv = b'T s in Rp:", print_element_hex_nnl(v, n, q);//
		short *K=new short[n];
		short h2=(1<<(logp-2))-(1<<(logp-logt-2)), log_h3=logp-logt-1;
		for(int kx=0;kx<n;++kx)
			K[kx]=(v[kx]-(c[kx]<<log_h3)+h2)>>(logp-1)&1;
		std::cout<<"\nK = v - (c<<ep-et-1) + 224 >> ep-1 in R2:", print_element_hex_nnl(K, n, q), std::cout<<endl;//
		unsigned char *KA=new unsigned char[size];
		saber_kdf(KA, K, n);
		delete[] v, K;
		delete[] b_dash, c;
		std::cout<<"KA:\t", print_buffer(KA, n>>3);
		std::cout<<"KB:\t", print_buffer(KB, n>>3);
		delete[] KA, KB;//destroy session keys
		delete[] s, seed_A, b;//destroy keys//*/
		
	/*	;
		//Pol mul test
		short n=64;
		short *a=new short[n], *s=new short[n], *b=new short[n];//b=a.s
		memset(b, 0, n*sizeof(short));
		for(int kv=0;kv<n;++kv)
		//	a[kv]=rand()&(q-1), s[kv]=rand()&(q-1);
		//	a[kv]=s[kv]=kv>=1&&kv<2;
		//	a[kv]=s[kv]=kv<3;
			a[kv]=1, s[kv]=1;
		std::cout<<"a:", print_element(a, n, q);//
		std::cout<<"s:", print_element(s, n, q);//

	//	toom_cook_4way((unsigned short*)a, (unsigned short*)s, (unsigned short*)b, 1<<logq, n);
	//	multiply_toom_cook4_saber(a, s, b, n, logq, anti_cyclic);
		multiply_karatsuba(a, s, b, n, logq, anti_cyclic);
		std::cout<<"b = a.s:", print_element(b, n, q);

		memset(b, 0, n*sizeof(short));
	//	multiply_toom_cook4_saber(a, s, b, n, logq, anti_cyclic);
	//	multiply_karatsuba(a, s, b, n, logq, anti_cyclic);
		multiply_polynomials_mod_powof2_add(a, s, b, n, logq, anti_cyclic);
		std::cout<<"b = a.s:", print_element(b, n, q);
	//	std::cout<<"\nb = a.s:", print_element_hex_nnl(b, n, q);//*/
		
		std::cout<<endl;
		if((_getch()&0xDF)=='X')
			break;
	}
#endif
	//Kyber 2017 PKC
#if 0
//	const short n=1024, q=12289, w=49,	sqrt_w=7,	beta_q=4091, beta_1=2304, q_1=-12287, n_1=12277, q_2=6145;//1024bit
//	const short n=512, q=12289, w=2401, sqrt_w=49, q_1=-12287, n_1=12277, beta_q=4091, beta_1=2304, barrett_k=27, barrett_m=10921;
	//X	const short n=512, q=19457, w=25, sqrt_w=5, q_1=-19455, n_1=19419, beta_q=7165, beta_1=5776;	//512bit
	const short n=256,	q=7681, w=3844, sqrt_w=62,	beta_q=4088, beta_1=900, q_1=-7679, n_1=7651, q_2=3841;//256bit
//	const short n=128,	q=769,	w=49,	sqrt_w=7,	beta_q=171,	beta_1=9,	q_1=-767,	n_1=763, q_2=385;
//X	const short n=128,	q=257,	w=9,	sqrt_w=3,	beta_q=1,	beta_1=1,	q_1=-255,	n_1=255, q_2=129;
//	const short n=64,	q=641,	w=441,	sqrt_w=21,	beta_q=154, beta_1=487, q_1=15745,	n_1=631, q_2=321, barrett_k=19; int barrett_m=568644;
//X	const short n=64,	q=257,	w=81,	sqrt_w=9,	beta_q=1,	beta_1=1,	q_1=-255,	n_1=253, q_2=129, barrett_k=15, barrett_m=127;//127.5
	int logn=log_2(n);
	bool anti_cyclic=true, forward_BRP=false;
	std::cout<<"Kyber PKC\t\tZ_"<<q<<"[x]/(x^"<<n<<"+1)\n\n";
	NTT_params p;
	number_transform_initialize(n, q, w, sqrt_w, anti_cyclic, p);
#ifdef XOF_USE_DRBG_AES128
	AES::initiate();
#endif
	const int align=sizeof(SIMD_type);
//	double sg=0, se=0, sd=0;
	long long tg=-1, te=-1, td=-1;
	for(int k_loop=1;;++k_loop)
//	for(int k_loop=1;k_loop<100;++k_loop)
//	for(;;)
	{
		//srand((int)__rdtsc());
	/*	//Kyber.AKE
		//B:
		Kyber_private_key pr_k2;
		Kyber_public_key pu_k2;
		kyber_generate(pr_k2, pu_k2, p);//B's static authentication key

		//A:
		Kyber_private_key pr_k1;
		Kyber_public_key pu_k1;
		kyber_generate(pr_k1, pu_k1, p);//A's static authentication key
		Kyber_private_key pr_k;
		Kyber_public_key pu_k;
		kyber_generate(pr_k, pu_k, p);
		Kyber_KE_ciphertext c2;
		int size=32;
		unsigned char *AK_buffer=new unsigned char[size*3];
		unsigned char *K_dash=AK_buffer, *K1_dash=AK_buffer+size, *K2=AK_buffer+size*2;
		kyber_encapsulate(pu_k2, c2, K2, p);

		//B:	<- pu_k, c2
		Kyber_KE_ciphertext c;
		unsigned char *BK_buffer=new unsigned char[size*3];
		unsigned char *K=BK_buffer, *K1=BK_buffer+size, *K2_dash=BK_buffer+size*2;
		kyber_encapsulate(pu_k, c, K, p);
		Kyber_KE_ciphertext c1;
		kyber_encapsulate(pu_k1, c1, K1, p);
		kyber_decapsulate(pr_k2, pu_k2, c2, K2_dash, p);
		unsigned char *B_key=new unsigned char[size];
		FIPS202_SHAKE128(BK_buffer, size*3, B_key, size);

		//A:	<- c, c1
		kyber_decapsulate(pr_k, pu_k, c, K_dash, p);
		kyber_decapsulate(pr_k1, pu_k1, c1, K1_dash, p);
		unsigned char *A_key=new unsigned char[size];
		FIPS202_SHAKE128(AK_buffer, size*3, A_key, size);

		std::cout<<"A's key: ", print_buffer(A_key, size);
		std::cout<<"B's key: ", print_buffer(B_key, size);

		delete[] AK_buffer, BK_buffer;
		delete[] A_key, B_key;//*/

	/*	//Kyber.UAKE
		//B:
		Kyber_private_key pr_k2;
		Kyber_public_key pu_k2;
		kyber_generate(pr_k2, pu_k2, p);
			
		//A:	<- pu_k2
		Kyber_private_key pr_k1;
		Kyber_public_key pu_k1;
		kyber_generate(pr_k1, pu_k1, p);
		Kyber_KE_ciphertext c2;
		int size=32;
		unsigned char *K1_dash=new unsigned char[size*2];
		unsigned char *K2=K1_dash+size;
		kyber_encapsulate(pu_k2, c2, K2, p);

		//B:	<- pu_k1, c2
		Kyber_KE_ciphertext c1;
		unsigned char *K1=new unsigned char[size*2];
		kyber_encapsulate(pu_k1, c1, K1, p);
		unsigned char *K2_dash=K1+size;
		bool B_success=kyber_decapsulate(pr_k2, pu_k2, c2, K2_dash, p);
		std::cout<<"\nB success = "<<B_success<<endl;
		unsigned char *key_B=new unsigned char[size];
		FIPS202_SHAKE128(K1, size*2, key_B, size);
		std::cout<<"K1:\t", print_buffer(K1, size);
		std::cout<<"K2':\t", print_buffer(K2_dash, size);
		std::cout<<"B's key: ", print_buffer(key_B, size);
		delete[] K1;

		//A:	<- c1
		bool A_success=kyber_decapsulate(pr_k1, pu_k1, c1, K1_dash, p);
		std::cout<<"\nA success = "<<A_success<<endl;
		unsigned char *key_A=new unsigned char[size];
		FIPS202_SHAKE128(K1_dash, size*2, key_A, size);
		std::cout<<"K1':\t", print_buffer(K1_dash, size);
		std::cout<<"K2:\t", print_buffer(K2, size);
		std::cout<<"A's key: ", print_buffer(key_A, size);
		delete[] K1_dash;//*/

	/*	//Kyber.CCA.KE
		Kyber_private_key pr_k;
		Kyber_public_key pu_k;
		long long t1=__rdtsc();
		kyber_generate(pr_k, pu_k, p);
		long long t_gen=__rdtsc()-t1;

		Kyber_KE_ciphertext ke_ct;
		int size=32;
		unsigned char *K=new unsigned char[size];
		t1=__rdtsc();
		kyber_encapsulate(pu_k, ke_ct, K, p);
		long long t_enc=__rdtsc()-t1;

		unsigned char *K2=new unsigned char[size];
		t1=__rdtsc();
		bool success=kyber_decapsulate(pr_k, pu_k, ke_ct, K2, p);
		long long t_dec=__rdtsc()-t1;

		std::cout<<"K1:\t", print_buffer(K, size);//
		std::cout<<"K2:\t", print_buffer(K2, size);//
		int error=0;
		std::cout<<"diff:\t";
		for(int k=0;k<32;++k)
		{
			printf("%02x", (int)K[k]^(int)K2[k]);
			error|=(int)K[k]^(int)K2[k];
		}
		std::cout<<"\nSuccess = "<<success<<endl;

		//sg+=t_gen, se+=t_enc, sd+=t_dec;
		//double tg=sg/k_loop, te=se/k_loop, td=sd/k_loop;
		//std::cout<<	"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		//std::cout<<	"Average:           "<<tg/freq<<"ms "<<tg<<", "<<te/freq<<"ms "<<te<<", "<<td/freq<<"ms "<<td<<",  "<<(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;

		delete[] K, K2;//*/

		//Kyber.CPA PKC
		Kyber_private_key pr_k;
		Kyber_public_key pu_k;
		long long t1=__rdtsc();
		kyber_generate(pr_k, pu_k, p);
		long long t_gen=__rdtsc()-t1;
			
		const char *message="12345678901234567890123456789012";//256bit
		std::cout<<"Message:\t"<<message<<endl;
	//	std::cout<<"m:\n", print_buffer(message, 32);//
		Kyber_ciphertext ct;
		t1=__rdtsc();
		kyber_encrypt(message, ct, pu_k, p);
		long long t_enc=__rdtsc()-t1;

		char message2[33]={0};
		t1=__rdtsc();
		kyber_decrypt(ct, message2, pr_k, p);
		long long t_dec=__rdtsc()-t1;
		std::cout<<"Decryption:\t"<<message2;
		bool error=false;
		for(int kx=0;kx<32;++kx)
			if(message2[kx]!=message[kx])
			{
				error=true;
				break;
			}
		if(error)
			std::cout<<"\tError\a";
	//	std::cout<<"m:\n", print_buffer(message2, 32);//
		_aligned_free(ct.u), _aligned_free(ct.v);//destroy ciphertext
		kyber_destroy(pr_k, pu_k, p);//destroy keys
		std::cout<<"\n\n";//*/

		//sg+=t_gen, se+=t_enc, sd+=t_dec;
		//double tg=sg/k_loop, te=se/k_loop, td=sd/k_loop;
		//std::cout<<	"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		//std::cout<<	"Average:           "<<tg/freq<<"ms "<<tg<<", "<<te/freq<<"ms "<<te<<", "<<td/freq<<"ms "<<td<<",  "<<(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		if(tg==-1||tg>t_gen)
			tg=t_gen;
		if(te==-1||te>t_enc)
			te=t_enc;
		if(td==-1||td>t_dec)
			td=t_dec;
		std::cout<<"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		std::cout<<"Minimum:           "<<double(tg)/freq<<"ms "<<tg<<", "<<double(te)/freq<<"ms "<<te<<", "<<double(td)/freq<<"ms "<<td<<",  "<<double(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		//std::cout<<"Generation:\t"<<double(t_gen)/freq<<"ms, "<<t_gen<<"cycles\n";//2.11015ms, 2740850c
		//std::cout<<"Encryption:\t"<<double(t_enc)/freq<<"ms, "<<t_enc<<"cycles\n";//2.21327ms, 2874790c
		//std::cout<<"Decryption:\t"<<double(t_dec)/freq<<"ms, "<<t_dec<<"cycles\n";//0.0660027ms, 85730c
		//std::cout<<"Total:\t"<<double(t_gen+t_enc+t_dec)/freq<<"ms, "<<(t_gen+t_enc+t_dec)<<"cycles\n";//4.38943ms, 5701370c

		std::cout<<endl;
		if((_getch()&0xDF)=='X')
			break;
	}
	number_transform_destroy(p);
#endif
	//NewHope'16	loop, API
#if 0
	const short n=1024, q=12289, w=49, sqrt_w=7,	q_1=-12287, n_1=12277, beta_q=4091, beta_1=2304, barrett_k=27, barrett_m=10921;
//	const short n=512, q=12289, w=2401, sqrt_w=49,	q_1=-12287, n_1=12277, beta_q=4091, beta_1=2304, barrett_k=27, barrett_m=10921;
//	const short n=256, q=7681, w=3844, sqrt_w=62,	q_1=-7679, n_1=7651, beta_q=4088, beta_1=900, barrett_k=21, barrett_m=273;
	bool anti_cyclic=true;
	std::cout<<"NewHope PKC\t\tZ_"<<q<<"[x]/(x^"<<n<<"+1)\n\n";
	NTT_params p;
	number_transform_initialize(n, q, w, sqrt_w, anti_cyclic, p);
#ifdef XOF_USE_DRBG_AES128
	AES::initiate();
#endif
//	const int align=sizeof(SIMD_type);
//	double sg=0, se=0, sd=0;
	long long tg=-1, te=-1, td=-1;
	for(int k_loop=1;;++k_loop)
//	for(int k_loop=1;k_loop<1000;++k_loop)
	{
		const int size=32;
		
	/*	//NewHope.CCA.KEM
		NewHope_KE_private_key ke_pr;
		NewHope_public_key pu_k;
		long long t1=__rdtsc();
		newhope_cca_generate(ke_pr, pu_k, p);
		long long t_gen=__rdtsc()-t1;

		NewHope_KE_ciphertext ke_ct;
		unsigned char K[size];
		t1=__rdtsc();
		newhope_cca_encapsulate(pu_k, ke_ct, K, p);
		long long t_enc=__rdtsc()-t1;

		unsigned char K2[size];
		t1=__rdtsc();
		newhope_cca_decapsulate(ke_pr, pu_k, ke_ct, K2, p);
		long long t_dec=__rdtsc()-t1;
		
		std::cout<<"K1:\t", print_buffer(K, size);//
		std::cout<<"K2:\t", print_buffer(K2, size);//
		int error=0;
		std::cout<<"diff:\t";
		for(int k=0;k<32;++k)
		{
			printf("%02x", (int)K[k]^(int)K2[k]);
			error|=(int)K[k]^(int)K2[k];
		}//*/

	/*	//NewHope.CPA.KEM
		NewHope_private_key k_pr;
		NewHope_public_key k_pu;
		long long t1=__rdtsc();
		newhope_generate(k_pr, k_pu, p);
		long long t_gen=__rdtsc()-t1;

		NewHope_ciphertext ct;
		unsigned char K[size];
		t1=__rdtsc();
		newhope_cpa_encapsulate(k_pu, ct, K, p);
		long long t_enc=__rdtsc()-t1;

		unsigned char K2[size];
		t1=__rdtsc();
		newhope_cpa_decapsulate(k_pr, k_pu, ct, K2, p);
		long long t_dec=__rdtsc()-t1;
		
		std::cout<<"K1:\t", print_buffer(K, size);//
		std::cout<<"K2:\t", print_buffer(K2, size);//
		int error=0;
		std::cout<<"diff:\t";
		for(int k=0;k<32;++k)
		{
			printf("%02x", (int)K[k]^(int)K2[k]);
			error|=(int)K[k]^(int)K2[k];
		}//*/

		//NewHope.CPA.PKC
		NewHope_private_key k_pr;
		NewHope_public_key k_pu;
		long long t1=__rdtsc();
		newhope_generate(k_pr, k_pu, p);
		long long t_gen=__rdtsc()-t1;

		const char *message="12345678901234567890123456789012";//256bit
	//	char message[size+1]={0};
	//	generate_uniform(size, (unsigned char*)message);
		unsigned char *e_seed=new unsigned char[size];
		generate_uniform(size, e_seed);
		NewHope_ciphertext ct;
		t1=__rdtsc();
		newhope_encrypt(message, ct, k_pu, e_seed, p);
		long long t_enc=__rdtsc()-t1;
		delete[] e_seed;

		char message2[33]={0};
		t1=__rdtsc();
		newhope_decrypt(ct, message2, k_pr, p);
		long long t_dec=__rdtsc()-t1;

		_aligned_free(ct.u_ntt), _aligned_free(ct.v_dash);//destroy ciphertext
		_aligned_free(k_pr.s_ntt), delete[] k_pu.seed, _aligned_free(k_pu.b_ntt);//destroy keys

		std::cout<<"Message:\t"<<message<<endl;
		std::cout<<"Decryption:\t"<<message2<<endl;
		bool error=false;
		for(int kx=0;kx<32;++kx)
			if(message2[kx]!=message[kx])
			{
				error=true;
				break;
			}
		if(error)
			std::cout<<"\tError\a";
	//	print_buffer(message, size);//
	//	print_buffer(message2, size);////*/
		
		std::cout<<endl;
	//	sg+=t_gen, se+=t_enc, sd+=t_dec;
	//	double tg=sg/k_loop, te=se/k_loop, td=sd/k_loop;
	//	std::cout<<	"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
	////	printf(		"Minimum:           %lfms %lf, %lfms %lf, %lfms %lf, %lfms %lf", tg/freq, tg, te/freq, te, td/freq, td, (tg+te+td)/freq, tg+te+td);
	//	std::cout<<	"Average:           "<<tg/freq<<"ms "<<tg<<", "<<te/freq<<"ms "<<te<<", "<<td/freq<<"ms "<<td<<",  "<<(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		if(tg==-1||tg>t_gen)
			tg=t_gen;
		if(te==-1||te>t_enc)
			te=t_enc;
		if(td==-1||td>t_dec)
			td=t_dec;
		std::cout<<"Gen,Enc,Dec,Total: "<<double(t_gen)/freq<<"ms "<<t_gen<<", "<<double(t_enc)/freq<<"ms "<<t_enc<<", "<<double(t_dec)/freq<<"ms "<<t_dec<<",  "<<double(t_gen+t_enc+t_dec)/freq<<"ms "<<(t_gen+t_enc+t_dec)<<endl;
		std::cout<<"Minimum:           "<<double(tg)/freq<<"ms "<<tg<<", "<<double(te)/freq<<"ms "<<te<<", "<<double(td)/freq<<"ms "<<td<<",  "<<double(tg+te+td)/freq<<"ms "<<(tg+te+td)<<endl;
		
	/*	const int size=32;
		unsigned char *seed=new unsigned char[size];//k_pu
		short *b_ntt=(short*)_aligned_malloc(n*sizeof(short), align);

		short *s_ntt=(short*)_aligned_malloc(n*sizeof(short), align);//k_pr

		{//Generate
			generate_uniform(size, k_pu.seed);
			short *a_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
			FIPS202_SHAKE128(k_pu.seed, size, (unsigned char*)a_ntt, n*sizeof(short));
		//	for(int k=0;k<n;++k)a_ntt[k]=2;//
		//	std::cout<<"a_ntt:", print_element(a_ntt, n, q);//

			short *s=(short*)_aligned_malloc(n*sizeof(short), align);
			newhope_generate_binomial_16(s, n);//
		//	for(int k=0;k<n;++k)s[k]=1;//

			apply_NTT(s, k_pr.s_ntt, p, false);
		//	std::cout<<"s_ntt:", print_element(s_ntt, n, q);//

			short *e=(short*)_aligned_malloc(n*sizeof(short), align);
			newhope_generate_binomial_16(e, n);//
		//	for(int k=0;k<n;++k)e[k]=1;//

			apply_NTT(e, k_pu.b_ntt, p, false);
		//	std::cout<<"e_ntt:", print_element(b_ntt, n, q);//

			multiply_ntt(b_ntt, a_ntt, k_pr.s_ntt, p);//b_ntt = a_ntt*s_ntt + e_ntt
		//	std::cout<<"b_ntt = a_ntt*s_ntt + e_ntt:", print_element(b_ntt, n, q);//
			_aligned_free(a_ntt), _aligned_free(s), _aligned_free(e);
		}

		const char *message="12345678901234567890123456789012";//256bit
		std::cout<<"Message:\t"<<message<<endl;
	//	unsigned char *e_seed=new unsigned char[size];
	//	generate_uniform(size, e_seed);
		short *u_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
		short *v_dash=(short*)_aligned_malloc(n*sizeof(short), align);
		{//Encrypt
			short *a_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
			FIPS202_SHAKE128(seed, size, (unsigned char*)a_ntt, n*sizeof(short));
		//	for(int k=0;k<n;++k)a_ntt[k]=2;//

			short *buffer=(short*)_aligned_malloc(3*n*sizeof(short), align),
				*s2=buffer, *e1=buffer+n, *e2=buffer+2*n;
		//	FIPS202_SHAKE128(e_seed, size, (unsigned char*)buffer, 3*n*sizeof(short));
			newhope_generate_binomial_16(s2, n);//
			newhope_generate_binomial_16(e1, n);//
			newhope_generate_binomial_16(e2, n);//
			//for(int k=0;k<n;++k)s2[k]=1;//
			//for(int k=0;k<n;++k)e1[k]=1;//
			//for(int k=0;k<n;++k)e2[k]=1;//

			short *s2_ntt=(short*)_aligned_malloc(n*sizeof(short), align);
			apply_NTT(s2, s2_ntt, p, false);

			apply_NTT(e1, u_ntt, p, false);//u_ntt = a_ntt*s2_ntt + e1_ntt
			//std::cout<<"e1_ntt:", print_element(u_ntt, n, q);//
			//std::cout<<"s2_ntt:", print_element(s2_ntt, n, q);//
			//std::cout<<"a_ntt:", print_element(a_ntt, n, q);//

			multiply_ntt(u_ntt, a_ntt, s2_ntt, p);
		//	std::cout<<"u_ntt = a_ntt*s2_ntt + e1_ntt:", print_element(u_ntt, n, q);//
			
			short *m=(short*)_aligned_malloc(n*sizeof(short), align);
			if(n==1024)
				for(int kb=0, q_2=(q>>1)+1;kb<256;++kb)
					m[kb]=m[kb+256]=m[kb+512]=m[kb+768] = q_2&-(message[kb>>3]>>(kb&7)&1);
			else if(n==512)
				for(int kb=0, q_2=(q>>1)+1;kb<256;++kb)
					m[kb]=m[kb+256]						= q_2&-(message[kb>>3]>>(kb&7)&1);
			else
				for(int kb=0, q_2=(q>>1)+1;kb<256;++kb)
					m[kb]								= q_2&-(message[kb>>3]>>(kb&7)&1);
			//for(int kc=0, q_2=(q>>1)+1;kc<size;++kc)
			//{
			//	for(int kb=0;kb<8;++kb)
			//	{
			//		int kx=(kc<<3)|kb;
			//		m[kx]=m[kx+256]=m[kx+512]=m[kx+768] = q_2&-(message[kc]>>kb&1);
			//	}
			//}
			//for(int kv=0, q_2=(q>>1)+1;kv<n;++kv)	//CRASH m: 256, n=1024
			//	m[kv]=q_2*(message[kv>>3]>>(kv&7)&1);
		//	std::cout<<"m:", print_element(m, n, q);//

			short *temp=(short*)_aligned_malloc(n*sizeof(short), align);
			memset(temp, 0, n*sizeof(short));
			multiply_ntt(temp, b_ntt, s2_ntt, p);//v' = INTT(b_ntt*s2_ntt) + e2 + m

			apply_inverse_NTT(temp, v_dash, p);

			for(int kx=0;kx<n;++kx)
			{
				auto &vk=v_dash[kx];
				vk+=e2[kx], vk-=q&-(vk>q);
				vk+=m[kx], vk-=q&-(vk>q);
			}
		//	std::cout<<"v' = INTT(b_ntt*s2_ntt) + e2 + m:", print_element(v_dash, n, q);//
			_aligned_free(a_ntt), _aligned_free(buffer), _aligned_free(s2_ntt), _aligned_free(m), _aligned_free(temp);
		}
		char message2[33]={0};
		{//Decrypt
			short *temp=(short*)_aligned_malloc(n*sizeof(short), align);
			memset(temp, 0, n*sizeof(short));
			multiply_ntt(temp, u_ntt, s_ntt, p);//m2 = v' - INTT(u_ntt*s_ntt)

			short *m2=(short*)_aligned_malloc(n*sizeof(short), align);
			apply_inverse_NTT(temp, m2, p);

			for(int kx=0;kx<n;++kx)
			{
				auto &vk=m2[kx];
				vk=v_dash[kx]-vk, vk+=q&-(vk<0);
			}
		//	std::cout<<"m2 = v' - INTT(u_ntt*s_ntt):", print_element(m2, n, q);//

			//if(n==1024)
			//	for(int kb=0, q_2=q>>1;kb<256;++kb)
			//	{
			//		int t=abs(m2[kb]-q_2)+abs(m2[kb+256]-q_2)+abs(m2[kb+512]-q_2)+abs(m2[kb+768]-q_2);
			//		t=(t-q)>>15;
			//		message2[kb>>3]|=t<<(kb&7);
			//	}
			//else if(n==512)
			//	for(int kb=0, q_2=q>>1;kb<256;++kb)
			//	{
			//		int t=abs(m2[kb]-q_2)+abs(m2[kb+256]-q_2);
			//		t=(t-q_2)>>15;
			//		message2[kb>>3]|=t<<(kb&7);
			//	}
			//else
				//for(int kb=0, q_2=q>>1;kb<256;++kb)
				//{
				//	int t=abs(m2[kb]-q_2);
				//	message2[kb>>3]|=t<<(kb&7);
				//}
				for(int kb=0, q_4=q/4, q3_4=q*3/4;kb<256;++kb)
				{
					int t=(m2[kb]>q_4)&(m2[kb]<q3_4);
					message2[kb>>3]|=t<<(kb&7);
				}
			_aligned_free(temp), _aligned_free(m2);
		}
		std::cout<<"Decryption:\t"<<message2;
		bool error=false;
		for(int kx=0;kx<32;++kx)
			if(message2[kx]!=message[kx])
			{
				error=true;
				break;
			}
		if(error)
			std::cout<<"\tError\a";
	//	print_buffer(message, size);//
	//	print_buffer(message2, size);////*/

		std::cout<<"\n\n";
		if((_getch()&0xDF)=='X')
			break;
	}
/*	{
	//	bitreverse_init();
		short n=1024, q=12289, w=49, sqrt_w=7, beta_q=4091, q_1=-12287;
	//	short n=4, q=17, w=4, sqrt_w=2;//n divisible by 4
		bool anti_cyclic=true;
		std::cout<<"NewHope\t\tZ_"<<q<<"[x]/(x^"<<n<<"+1)\n\n";
		int logn=log_2(n);
		int n_1=0;
		if(!extended_euclidean_algorithm(n, q, n_1))
			std::cout<<n<<" has no inverse mod "<<q<<endl;
		else
		{
			NTT_params p;
			number_transform_initialize(n, q, w, sqrt_w, anti_cyclic, p);
			
			QueryPerformanceCounter(&li);
			ticks=li.QuadPart;
			const int align=sizeof(SIMD_type);
			short *S_s=(short*)_aligned_malloc(n*sizeof(short), align);//server secret
			newhope_generate_binomial_16(S_s, n);
			make_small(S_s, n, q, q_1);
		//	print_element(S_s, n);

			short *C_s=(short*)_aligned_malloc(n*sizeof(short), align);//client secret
			newhope_generate_binomial_16(C_s, n);
			make_small(C_s, n, q, q_1);
		//	print_element(S_s, n);
			for(;;)
			{
				long long t1=__rdtsc();
			//	QueryPerformanceCounter(&li);
			//	ticks=li.QuadPart;

				//Alice (server)
				int seed_size=32;
				unsigned char *S_seed=(unsigned char*)_aligned_malloc(seed_size*sizeof(unsigned char), align);
				generate_uniform(seed_size*sizeof(unsigned char), (unsigned char*)S_seed);
				short *S_a_hat=(short*)_aligned_malloc(n*sizeof(short), align);//a^ = Parse(SHAKE-128(seed))
#ifdef PROFILING_ALG
				memset(S_a_hat, 0, n*sizeof(short));
#else if
				FIPS202_SHAKE128(S_seed, seed_size*sizeof(unsigned char), (unsigned char*)S_a_hat, n*sizeof(short));
#endif
				//std::cout<<"Server a_hat :\n", print_table_NH(S_a_hat, n), std::cout<<endl;//
				short *S_e=(short*)_aligned_malloc(n*sizeof(short), align);					//s, e from psi16^n
				newhope_generate_binomial_16(S_e, n);
			//	memset(S_e, 0, n*sizeof(short));//
				short *S_s_hat=(short*)_aligned_malloc(n*sizeof(short), align);
				//std::cout<<"Client s =\n", print_table_NH(S_s, n), std::cout<<endl;//
				apply_NTT(S_s, S_s_hat, p);
			//	make_small(S_s, n, q, q_1);
				//number_transform(S_s, S_s_hat, roots, q, logn, n);						//s^ = NTT(s)
				//number_transform(S_s_hat, S_s, iroots, q, logn, n);//
				//for(int k=0;k<n;++k)
				//	S_s[k]*=n_1, S_s[k]%=q;
				//if(anti_cyclic)
				//{
				//	for(int k=n-1, factor=sqrt_w;k>0;--k)
				//	{
				//		S_s[k]*=-factor, S_s[k]%=q, S_s[k]+=q&-(c_ntt[k]<0);
				//		factor*=sqrt_w, factor%=q, factor+=q&-(factor<0);
				//	}
				//}
				short *S_b_hat=(short*)_aligned_malloc(n*sizeof(short), align);
				apply_NTT(S_e, S_b_hat, p);			//b^ = a^ * s^ + NTT(e)
				multiply_ntt(S_b_hat, S_a_hat, S_s_hat, p);
				//std::cout<<"Server b^ = a^ * s^ + NTT(e) :\n", print_table_NH(S_b_hat, n), std::cout<<endl;//
				//ma = encodeA(seed, b^)

				//Bob		client receives ma=encodeA(seed, b_hat)
				short *C_e=(short*)_aligned_malloc(n*sizeof(short), align),
					*C_e2=(short*)_aligned_malloc(n*sizeof(short), align);
				newhope_generate_binomial_16(C_e, n);//s', e', e'' from psi16^n
				newhope_generate_binomial_16(C_e2, n);
			//	memset(C_e, 0, n*sizeof(short));//
			//	memset(C_e2, 0, n*sizeof(short));//
				short *C_b_hat=(short*)_aligned_malloc(n*sizeof(short), align);				//(b^, seed) = decodeA(ma)		{seed, b_hat} = {32, 2048->1792} = 1824 bytes
				unsigned char *C_seed=(unsigned char*)_aligned_malloc(seed_size*sizeof(unsigned char), align);
				memcpy(C_b_hat, S_b_hat, n*sizeof(short));
				memcpy(C_seed, S_seed, seed_size*sizeof(unsigned char));
				short *C_a_hat=(short*)_aligned_malloc(n*sizeof(short), align);				//a^ = Parse(SHAKE-128(seed))
#ifdef PROFILING_ALG
				memset(C_a_hat, 0, n*sizeof(short));
#else if
				FIPS202_SHAKE128(C_seed, seed_size*sizeof(unsigned char), (unsigned char*)C_a_hat, n*sizeof(short));
#endif
				//std::cout<<"Client a^ :\n", print_table_NH(S_a_hat), std::cout<<endl;//
				short *C_t_hat=(short*)_aligned_malloc(n*sizeof(short), align);
				apply_NTT(C_s, C_t_hat, p);			//t^ = NTT(s')
				short *C_u_hat=(short*)_aligned_malloc(n*sizeof(short), align);
				apply_NTT(C_e, C_u_hat, p);			//u^ = a^ * t^ + NTT(e')
				multiply_ntt(C_u_hat, C_a_hat, C_t_hat, p);
				//std::cout<<"Client u^ = a^ * NTT(s) + NTT(e'):\n", print_table_NH(C_u_hat), std::cout<<endl;//
				multiply_ntt(C_b_hat, C_b_hat, C_t_hat, p);							//v = NTT-1(b^ * t^) + e''
				short *C_v=(short*)_aligned_malloc(n*sizeof(short), align);
				apply_inverse_NTT(C_b_hat, C_v, p);
#ifndef PROFILING_ALG
	//			std::cout<<"Client v = NTT-1(b^ * NTT(C_s)) =\n", print_table_NH(C_v, n), std::cout<<endl;//
#endif
				unsigned char *C_r=(unsigned char*)_aligned_malloc(n*sizeof(unsigned char), align);//mod 4
				NewHope_HelpRec(C_v, C_r);
				//mb = encodeB(u_hat, r)		{r[1024] mod 4, u_hat} = {256, 2048->1792} = 2048 bytes
				unsigned char *C_nu=(unsigned char*)_aligned_malloc(n/4*sizeof(unsigned char), align);//256bit
				NewHope_Rec(C_v, C_r, C_nu);
#ifndef PROFILING_ALG
				unsigned char *C_key=(unsigned char*)_aligned_malloc(n/4*sizeof(unsigned char), align);
				FIPS202_SHA3_256(C_nu, 256, C_key);
#endif

				//Alice			server receives mb = encodeB(u_hat, r)
				short *S_u_hat=(short*)_aligned_malloc(n*sizeof(short), align);
				unsigned char *S_r=(unsigned char*)_aligned_malloc(n*sizeof(unsigned char), align);
				memcpy(S_u_hat, C_u_hat, n*sizeof(short));
				memcpy(S_r, C_r, n*sizeof(unsigned char));
				multiply_ntt(S_u_hat, S_u_hat, S_s_hat, p);							//v' = NTT-1(u^ * s^)
				short *S_v_dash=(short*)_aligned_malloc(n*sizeof(short), align);
				apply_inverse_NTT(S_u_hat, S_v_dash, p);
	//			std::cout<<"Server v' = NTT-1((a^ * NTT(C_s)) * S_s^) =\n", print_table_NH(S_v_dash, n), std::cout<<endl;//
				unsigned char *S_nu=(unsigned char*)_aligned_malloc(n/4*sizeof(unsigned char), align);
				NewHope_Rec(S_v_dash, S_r, S_nu);
				
				//i7-6800K SSE: 0.397158ms 1318696c
				//U7700 SSE:	1.51805ms 2157450c
				long long t2=__rdtsc();
				std::cout<<double(t2-t1)/freq<<" ms, "<<(t2-t1)<<" cycles\n";
			//	QueryPerformanceCounter(&li);
			//	std::cout<<(1000.*(li.QuadPart-ticks)/freq)<<" ms\n";
#ifndef PROFILING_ALG
				unsigned char *S_key=(unsigned char*)_aligned_malloc(n/4*sizeof(unsigned char), align);
				FIPS202_SHA3_256(S_nu, 256, S_key);

				std::cout<<"Client key: ";
				for(int k=0;k<32;++k)
					printf("%02x", (int)C_key[k]);
				std::cout<<"\nServer key: ";
				for(int k=0;k<32;++k)
					printf("%02x", (int)S_key[k]);
				int error=0;
				std::cout<<"\nDifference: ";
				for(int k=0;k<32;++k)
				{
					printf("%02x", (int)C_key[k]^(int)S_key[k]);
					error|=(int)C_key[k]^(int)S_key[k];
				}
				if(error)
					std::cout<<"\t<-";
				std::cout<<"\n\n";
				_aligned_free(C_key), _aligned_free(S_key);
			//	delete[] C_key, S_key;
#endif
				_aligned_free(S_seed), _aligned_free(S_a_hat), _aligned_free(S_e), _aligned_free(S_s_hat), _aligned_free(S_b_hat),
					_aligned_free(S_u_hat), _aligned_free(S_r), _aligned_free(S_v_dash), _aligned_free(S_nu);
				_aligned_free(C_e), _aligned_free(C_b_hat), _aligned_free(C_seed), _aligned_free(C_a_hat), _aligned_free(C_t_hat);
					_aligned_free(C_v), _aligned_free(C_r), _aligned_free(C_nu);
			//	delete[]
			//		S_seed, S_a_hat, S_e, S_s_hat, S_b_hat,	S_u_hat, S_r, S_v_dash, S_nu,
			//		C_e, C_e2, C_b_hat, C_seed, C_a_hat, C_t_hat, C_u_hat, C_v, C_r, C_nu;

				if((_getch()&0xDF)=='X')
					break;
			}
			_aligned_free(S_s), _aligned_free(C_s);
			number_transform_destroy(p);
		//	delete[] bitreverse_table;
		//	delete[] roots, iroots, S_s, C_s;
		//	delete[] roots, iroots,
		//		S_seed, S_a_hat, S_s, S_e, S_s_hat, S_b_hat,	S_u_hat, S_r, S_v_dash, S_nu, S_key,
		//		C_s, C_e, C_e2, C_b_hat, C_seed, C_a_hat, C_t_hat, C_u_hat, C_v, C_r, C_nu, C_key;
		}
	}//*/
#endif
	
/*	//LP11
	printf("LP11 LWE\nn1=%d, n2=%d, l=%d, q=%d, s=%g\n\n\n", LP11_LWE::n1, LP11_LWE::n2, LP11_LWE::l, LP11_LWE::q, LP11_LWE::s);
//	printf("LP11 LWE\n\n\n");
	LP11_LWE::Generate();
	char message[17]="- sample text --";
//	char message[17]="0123456789ABCDEF";
	printf("Input message: %s\n", message);
	for(;;)
	{
	//	int *c1=0, *c2=0;
		int c1[LP11_LWE::n2]={0}, c2[LP11_LWE::l]={0};
	//	LP11_LWE::Encrypt((unsigned char*)message, c1, c2);
		LP11_LWE::Encrypt((unsigned char*)message, (int*)c1, (int*)c2);
		for(int k=0;k<LP11_LWE::n2;++k)//c1: 2 LSB discarded
			c1[k]>>=2, c1[k]<<=2;
		for(int k=0;k<LP11_LWE::l;++k)//c2: 7 LSB discarded
			c2[k]>>=7, c2[k]<<=7;
	//	char *m=new char[16];
		char m[17]={0};
		LP11_LWE::Decrypt(c1, c2, (unsigned char*)m);
		m[16]='\0';
		printf("Decrypted message:\n\t%s\n\n", m);
	//	delete[] m;
		printf("\nInput message (16 characters): ");

		std::cin.getline(message, 17);
		//std::string str;
		//std::cin>>str;//no spaces
		//if(str.size()<16)
		//{
		//	for(int k=0, kEnd=str.size();k<kEnd;++k)
		//		message[k]=str[k];
		//	for(int k=str.size();k<16;++k)
		//		message[k]=0;
		//}
		//else
		//{
		//	for(int k=0;k<16;++k)
		//		message[k]=str[k];
		//}
		message[16]='\0';
	}
	LP11_LWE::Destroy();//*/

/*	//LPR10-LWE		"On ideal lattices and ring-LWE" - V Lyubashevsky, C Peikert, and O Regev
	printf("LPR10-LWE\t\tZ_%d[x]/(x^%d-1), s=%g\n\n\n", v2_q, v2_N, v2_s);			//LPR10-LWE				"small" vs Xi_s?
//	printf("LPR10-LWE\nn=%d, q=%d, s=%g\n\n\n", v2_N, v2_q, v2_s);
//	printf("LPR10-LWE\n\n\n");
	Zq_xn_1 e, a, s;
	//generation
	e.generate_small(v2_s), a.generate_rand(), s.generate_small(v2_s);//s private key
	Zq_xn_1 b=a*s+e;//(a, b) public key
	int const message_size=40;
	char message[message_size+1]="deceptive deceptive deceptive deceptive.";
//	char message[message_size+1]="captain captain captain captain captain.";
//	char message[message_size+1]="0123456789012345678901234567890123456789";
	printf("Input message: %s\n", message);
	for(;;)
	{
		//encryption: generate buffers
		Zq_xn_1 e1, e2, t;
		e1.generate_small(v2_s), e2.generate_small(v2_s), t.generate_small(v2_s);
		Zq_xn_1 c1=a*t+e1, c2=b*t+e2;
		//encryption: receive stream
		Zq_xn_1 encode_m;
	//	for(int k=0;k<v2_N;++k)//N bit message
	//		encode_m.v[k]=k%2;
		//	encode_m.v[k]=rand()%2;
	//	encode_m.print_table("m");
		int m_amp=v2_q/2;
		for(int k=0;k<v2_N;++k)//encode(m)
			encode_m.v[k]=m_amp&-((message[k/8]>>k%8)&1);
		//	encode_m.v[k]*=m_amp;
		c2+=encode_m;
	//	c1.print_table("\n\nc1 = a*t + e1");
	//	c2.print_table("\n\nc2 = b*t + e2 + encode(m)");
		for(int k=0;k<v2_N;++k)//[PG13]: 7 least significant bits are unnecessary
			c2.v[k]>>=7, c2.v[k]<<=7;

		//decryption
		Zq_xn_1 plain=c2-c1*s;
		plain.make_small2();
		plain.print_table("\n\np = c2 - c1*s");
		int threshold=v2_q/4;
	//	for(int k=0;k<v2_N;++k)
	//		plain.v[k]=abs(plain.v[k])>threshold;
	//	plain.print_table("\n\nplain = decode(p)");
		memset(message, 0, (message_size+1)*sizeof(unsigned char));
		for(int k=0;k<v2_N;++k)
			message[k/8]|=(abs(plain.v[k])>threshold)<<(k%8);
		printf("Decrypted message:\n\t%s\n\n", message);
		printf("\nInput message (16 characters): ");
		std::cin.getline(message, message_size+1);
		message[message_size]='\0';
	}//*/
	//small range		abs(p) ranges						separation		q/4 = 147730.25
	//[-q/2, q/2]		0:[2, 28293]	1:[261963, 295430]	145128
	//[-1, 1]			0:[0, 47]		1:[295403, 295460]	147725
	//{-1, 1}			0:[1, 73]		1:[295393, 295460]	147733
	//int min0, max0, min1, max1;//find error ranges of 0 and 1
	//bool uninit0=true, uninit1=true;
	//for(int k=0;k<v2_N&&uninit0|uninit1;++k)
	//{
	//	int abs_vk=abs(plain.v[k]);
	//	if(encode_m.v[k])
	//	{
	//		min1=max1=abs_vk;
	//		uninit1=false;
	//	}
	//	else
	//	{
	//		min0=max0=abs_vk;
	//		uninit0=false;
	//	}
	//}
	//for(int k=0;k<v2_N;++k)
	//{
	//	int abs_vk=abs(plain.v[k]);
	//	if(encode_m.v[k])
	//	{
	//		if(min1>abs_vk)
	//			min1=abs_vk;
	//		if(max1<abs_vk)
	//			max1=abs_vk;
	//	}
	//	else
	//	{
	//		if(min0>abs_vk)
	//			min0=abs_vk;
	//		if(max0<abs_vk)
	//			max0=abs_vk;
	//	}
	//}
	//std::cout<<"\n0 range:\t"<<min0<<"\t"<<max0<<"\n"
	//	"1 range:\t"<<min1<<"\t"<<max1<<"\n";//*/

	
/*	//LP10	2010-11-30 "Better key sizes (and attacks) for LWE-based encryption" - R Lindner and C Peikert
	printf("LP10 ring-LWE\nn=%d, q=%d, s=%g\n\n\n", v2_N, v2_q, v2_s);			//LP10 ring-LWE
//	printf("LP10 ring-LWE\n\n\n");
	Zq_xn_1 r1, r2, a;
	//generation
	r1.generate_small(v2_s), r2.generate_small(v2_s), a.generate_rand();//r2: private
	Zq_xn_1 b=r1-a*r2;//(a, b): public
	
	int const message_size=40;
	char message[message_size+1]="deceptive deceptive deceptive deceptive.";
//	char message[message_size+1]="captain captain captain captain captain.";
//	char message[message_size+1]="0123456789012345678901234567890123456789";
	printf("Input message: %s\n", message);
	for(;;)
	{
		//encryption: ready buffers
		Zq_xn_1 e1, e2, e3;
		e1.generate_small(v2_s), e2.generate_small(v2_s), e3.generate_small(v2_s);
		Zq_xn_1 c1=a*e1+e2, c2=b*e1+e3;
		//encryption: receive stream
		Zq_xn_1 encode_m;
	//	for(int k=0;k<v2_N;++k)//N bit message
		//	encode_m.v[k]=rand()%2;
		//	encode_m.v[k]=k%2;
	//	encode_m.print_table("m");
		int m_amp=v2_q/2;
		for(int k=0;k<v2_N;++k)//encode(m)
			encode_m.v[k]=m_amp&-((message[k/8]>>k%8)&1);
		//	encode_m.v[k]*=m_amp;
	//	encode_m.print_table("\n\nencode(m)");
		c2+=encode_m;
		for(int k=0;k<v2_N;++k)//[PG13]: 7 least significant bits are unnecessary
			c2.v[k]>>=7, c2.v[k]<<=7;
		c1.print_table("\n\nc1 = a*e1 + e2");
		c2.print_table("\n\nc2 = b*e1 + e3 + encode(m)");

		//decryption
		Zq_xn_1 plain=c1*r2+c2;
		plain.make_small2();
	//	plain.print_table("\n\np = c1*r2 + c2");
		int threshold=v2_q/4;
		memset(message, 0, (message_size+1)*sizeof(unsigned char));
		for(int k=0;k<v2_N;++k)
			message[k/8]|=(abs(plain.v[k])>threshold)<<(k%8);
		printf("Decrypted message:\n\t%s\n\n", message);
		printf("\nInput message (16 characters): ");
		std::cin.getline(message, message_size+1);
		message[message_size]='\0';
		//small range		abs(p) ranges						separation		q/4 = 147730.25
		//{-q/2, q/2}		0:[249, 29254]  1:[270943, 295304]	150098.5
		//[-1, 1]			0:[0, 47]		1:[295401, 295460]	147724
		//{-1, 1}			0:[1, 61]		1:[295391, 295460]	147726
		//int min0, max0, min1, max1;//find error ranges of 0 and 1
		//bool uninit0=true, uninit1=true;
		//for(int k=0;k<v2_N&&uninit0|uninit1;++k)
		//{
		//	int abs_vk=abs(plain.v[k]);
		//	if(encode_m.v[k])
		//	{
		//		min1=max1=abs_vk;
		//		uninit1=false;
		//	}
		//	else
		//	{
		//		min0=max0=abs_vk;
		//		uninit0=false;
		//	}
		//}
		//for(int k=0;k<v2_N;++k)
		//{
		//	int abs_vk=abs(plain.v[k]);
		//	if(encode_m.v[k])
		//	{
		//		if(min1>abs_vk)
		//			min1=abs_vk;
		//		if(max1<abs_vk)
		//			max1=abs_vk;
		//	}
		//	else
		//	{
		//		if(min0>abs_vk)
		//			min0=abs_vk;
		//		if(max0<abs_vk)
		//			max0=abs_vk;
		//	}
		//}
		//std::cout<<"\n0 range:\t"<<min0<<"\t"<<max0<<"\n"
		//	"1 range:\t"<<min1<<"\t"<<max1<<"\n";
		//for(int k=0;k<v2_N;++k)//decode(p)
		//	plain.v[k]=abs(plain.v[k])>150099;
		//plain.print_table("\n\nplain = decode(p)");
	}//*/
	if(!use_rand)
	{
		if(!CryptReleaseContext(hProv, 0))
		{
			int error=GetLastError();
			std::cout<<"Error CryptReleaseContext(): "<<error;
			_getch();
		}
	}
}
