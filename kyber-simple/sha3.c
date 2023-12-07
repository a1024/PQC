//SHA-3
//http://keccak.noekeon.org/tweetfips202.html
static inline unsigned long long load64(const unsigned char *x)
{
	unsigned i;
	long long u=0;
	for(i=0; i<8; ++i)
		u<<=8, u|=x[7-i];
	return u;
}
static inline void store64(unsigned char *x, unsigned long long u)
{
	unsigned i;
	for(i=0; i<8; ++i)
		x[i]=u, u>>=8;
}
static inline void xor64(unsigned char *x, unsigned long long u)
{
	unsigned i;
	for(i=0; i<8; ++i)
		x[i]^=u, u>>=8;
}
int			LFSR86540(unsigned char *R)
{
	int result=*R&1;
	*R=(*R<<1)^(*R&0x80?0x71:0);
	return result;
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
			C[x]=load64((unsigned char*)s+8*(x+5*0))
				^load64((unsigned char*)s+8*(x+5*1))
				^load64((unsigned char*)s+8*(x+5*2))
				^load64((unsigned char*)s+8*(x+5*3))
				^load64((unsigned char*)s+8*(x+5*4));
		}
		for(x=0;x<5;++x)
		{
			D=C[(x+4)%5]^(((C[(x+1)%5])<<1)^((C[(x+1)%5])>>(64-1)));
			for(y=0;y<5;++y)
				xor64((unsigned char*)s+8*(x+5*y),D);
		}
		x=1, y=r=0;		//rho*pi
		D=load64((unsigned char*)s+8*(x+5*y));
		for(j=0;j<24;++j)
		{
			r+=j+1;
			Y=(2*x+3*y)%5, x=y, y=Y;
			C[0]=load64((unsigned char*)s+8*(x+5*y));
			store64((unsigned char*)s+8*(x+5*y), ((D<<r%64)^(D>>(64-r%64))));
			D=C[0];
		}
		for(y=0;y<5;++y)	//chi
		{
			for(x=0;x<5;++x)
				C[x]=load64((unsigned char*)s+8*(x+5*y));
			for(x=0;x<5;++x)
				store64((unsigned char*)s+8*(x+5*y), C[x]^((~C[(x+1)%5])&C[(x+2)%5]));
		}
		for(j=0;j<7;++j)	//iota
		{
			if(LFSR86540(&R))
				xor64((unsigned char*)s+8*(0 +5*0), (unsigned long long)1<<((1<<j)-1));
		}
	}
}
void Keccak(unsigned r, unsigned c, const unsigned char *in, unsigned long long inLen, unsigned char sfx, unsigned char *out, unsigned long long outLen)
{
	__declspec(align(8)) unsigned char s[200]={0};//initialize
	unsigned R=r/8, i, b=0;
	memset(s, 0, 200);
//	for(i=0; i<200; ++i)
//		s[i]=0;
	while(inLen>0)		//absorb
	{
		b=inLen<R?(unsigned)inLen:R;
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
		b=outLen<R?(unsigned)outLen:R;
		for(i=0;i<b;++i)
			out[i]=s[i];
		out+=b, outLen-=b;
		if(outLen>0)
			KeccakF1600(s);
	}
}