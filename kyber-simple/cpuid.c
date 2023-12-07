#include"pqc.h"
#include<stdio.h>
#include<string.h>
#ifdef _MSC_VER
#include<intrin.h>
#elif defined __GNUC__
#include<x86intrin.h>
#endif

void get_cpuinfo(CPUInfo *info)
{
	//https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=msvc-170
	//https://en.wikipedia.org/wiki/CPUID?useskin=monobook
	int regs[4];
	int nids, nexids;
	int f1_ecx, f1_edx;
	int f7_ebx, f7_ecx;

	__cpuid(regs, 0);
	nids=regs[0];
	
	__cpuidex(regs, 0, 0);
	memcpy(info->vendor, regs+1, 4);
	memcpy(info->vendor+4, regs+3, 4);
	memcpy(info->vendor+8, regs+2, 4);
	memset(info->vendor+12, 0, 4);

	if(nids>1)
	{
		__cpuidex(regs, 1, 0);
		f1_ecx=regs[2], f1_edx=regs[3];
	}
	else
		f1_ecx=0, f1_edx=0;

	if(nids>7)
	{
		__cpuidex(regs, 7, 0);
		f7_ebx=regs[1], f7_ecx=regs[2];
	}
	else
		f7_ebx=0, f7_ecx=0;

	__cpuid(regs, 0x80000000);
	nexids=regs[0];
	if(nexids>=0x80000004)
	{
		__cpuidex(regs, 0x80000002, 0);
		memcpy(info->brand, regs, sizeof(regs));
		__cpuidex(regs, 0x80000003, 0);
		memcpy(info->brand+sizeof(regs), regs, sizeof(regs));
		__cpuidex(regs, 0x80000004, 0);
		memcpy(info->brand+sizeof(regs)*2, regs, sizeof(regs));
		memset(info->brand+sizeof(regs)*3, 0, sizeof(regs));//for struct alignment
	}
	else
		memset(info->brand, 0, sizeof(info->brand));

	info->mmx=f1_edx>>23&1;
	info->sse=f1_edx>>25&1;
	info->sse2=f1_edx>>26&1;
	info->sse3=f1_ecx&1;
	info->ssse3=f1_ecx>>9&1;
	info->sse4_1=f1_ecx>>19&1;
	info->sse4_2=f1_ecx>>20&1;
	info->fma=f1_ecx>>12&1;
	info->aes=f1_ecx>>25&1;
	info->sha=f7_ebx>>29&1;
	info->avx=f1_ecx>>28&1;
	info->avx2=f7_ebx>>5&1;
	info->avx512F=f7_ebx>>16&1;
	info->avx512PF=f7_ebx>>26&1;
	info->avx512ER=f7_ebx>>27&1;
	info->avx512CD=f7_ebx>>28&1;
	info->f16c=f1_ecx>>29&1;
	info->rdrand=f1_ecx>>30&1;
	info->rdseed=f7_ebx>>18&1;
}
void print_cpuinfo(CPUInfo *info)
{
	printf("CPU: %s\n", info->brand);
	if(info->mmx)printf(" MMX");
	if(info->sse)printf(" SSE");
	if(info->sse2)printf(" SSE2");
	if(info->sse3)printf(" SSE3");
	if(info->ssse3)printf(" SSSE3");
	if(info->sse4_1)printf(" SSSE4.1");
	if(info->sse4_2)printf(" SSSE4.2");
	if(info->fma)printf(" FMA");
	//printf(" ");
	if(info->aes)printf(" AES");
	if(info->sha)printf(" SHA");
	//printf(" ");
	if(info->avx)printf(" AVX");
	if(info->avx2)printf(" AVX2");
	//printf(" ");
	if(info->avx512F)printf(" AVX512F");
	if(info->avx512PF)printf(" AVX512PF");
	if(info->avx512ER)printf(" AVX512ER");
	if(info->avx512CD)printf(" AVX512CD");
	//printf(" ");
	if(info->f16c)printf(" F16C");
	if(info->rdrand)printf(" RDRAND");
	if(info->rdseed)printf(" RDSEED");
	printf("\n\n");
}