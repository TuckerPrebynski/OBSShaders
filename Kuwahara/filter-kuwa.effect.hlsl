// OBS-specific syntax adaptation to HLSL standard to avoid errors reported by the code editor
#define SamplerState sampler_state
#define Texture2D texture2d

// Uniform variables set by OBS (required)
uniform float4x4 ViewProj; // View-projection matrix used in the vertex shader
uniform Texture2D image;   // Texture containing the source picture

// Constants
#define PI 3.141592653589793238


// General properties
uniform int box_radius = 4.0;
uniform int sharpness = 8;
//uniform int box_area = box_radius*box_radius;


// Size of the source picture
uniform int width;
uniform int height;

SamplerState linear_wrap
{
    Filter    = Linear; 
    AddressU  = Wrap;
    AddressV  = Wrap;
};
// Interpolation method and wrap mode for sampling a texture
SamplerState linear_clamp
{
    Filter    = Linear;     // Anisotropy / Point / Linear
    AddressU  = Clamp;      // Wrap / Clamp / Mirror / Border / MirrorOnce
    AddressV  = Clamp;      // Wrap / Clamp / Mirror / Border / MirrorOnce
    BorderColor = 00000000; // Used only with Border edges (optional)
};

SamplerState point_clamp
{
    Filter    = Point; 
    AddressU  = Clamp;
    AddressV  = Clamp;
};
// Data type of the input of the vertex shader
struct vertex_data
{
    float4 pos : POSITION;  // Homogeneous space coordinates XYZW
    float2 uv  : TEXCOORD0; // UV coordinates in the source picture
};

// Data type of the output returned by the vertex shader, and used as input 
// for the pixel shader after interpolation for each pixel
struct pixel_data
{
    float4 pos : POSITION;  // Homogeneous screen coordinates XYZW
    float2 uv  : TEXCOORD0; // UV coordinates in the source picture
};

// Vertex shader used to compute position of rendered pixels and pass UV
pixel_data vertex_shader_kuwa(vertex_data vertex)
{
    pixel_data pixel;
    pixel.pos = mul(float4(vertex.pos.xyz, 1.0), ViewProj);
    pixel.uv  = vertex.uv;
    return pixel;
}




//RGB to HCV and back jacked from https://www.chilliant.com/rgb2hsv.html
float Epsilon = .0000000001;
float3 HUEtoRGB(in float H)
  {
    float R = abs(H * 6 - 3) - 1;
    float G = 2 - abs(H * 6 - 2);
    float B = 2 - abs(H * 6 - 4);
    return saturate(float3(R,G,B));
  }
  
float3 HSVtoRGB(in float3 HSV)
  {
    float3 RGB = HUEtoRGB(HSV.x);
    return ((RGB - 1) * HSV.y + 1) * HSV.z;
  }
float3 RGBtoHCV(in float3 RGB)
  {
    // Based on work by Sam Hocevar and Emil Persson
    float4 P = (RGB.g < RGB.b) ? float4(RGB.bg, -1.0, 2.0/3.0) : float4(RGB.gb, 0.0, -1.0/3.0);
    float4 Q = (RGB.r < P.x) ? float4(P.xyw, RGB.r) : float4(RGB.r, P.yzx);
    float C = Q.x - min(Q.w, Q.y);
    float H = abs((Q.w - Q.y) / (6 * C + Epsilon) + Q.z);
    return float3(H, C, Q.x);
  }
float3 RGBtoHSV(in float3 RGB)
  {
    float3 HCV = RGBtoHCV(RGB);
    float S = HCV.y / (HCV.z + Epsilon);
    return float3(HCV.x, S, HCV.z);
  }
 
//Refactored python code of welford's online algorithm
//if it dont work blame https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance 
float4 welford_update(in float4 existing_aggregate)
{

   //count map x
	//mean map y
	//M2 map z
	//new_value map w
    // count += 1
	float delta;
	float delta2;
	float M2;
	float mean;
	float1 count = existing_aggregate.x + 1.0;
	
	//delta = new_value - mean
    delta = (existing_aggregate.w - existing_aggregate.y);
	
    //mean += delta / count
	mean = existing_aggregate.y + (delta/count);
	
    //delta2 = new_value - mean
	delta2 = (existing_aggregate.w - mean);
    
	//M2 += delta * delta2
	M2 = (existing_aggregate.z + (delta * delta2));
    
	//return float4(count, mean, M2, 1.0)
	return float4(count,mean,M2, 1.0);
}

// Retrieve the mean, variance and sample variance from an aggregate
float4 welford_finalize(in float4 existing_aggregate)
{
    //count map x
	//mean map y
	//M2 map z
	//(count, mean, M2) = existing_aggregate
    //nixed if statment b/c count should never be less than four
	//if count < 2:
    //    return float("nan")
    //else:
	//	(mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
    //    return (mean, variance, sample_variance)
    return float4(existing_aggregate.y, sqrt(existing_aggregate.z / float(existing_aggregate.x)), sqrt(existing_aggregate.z / float(existing_aggregate.x-1.0)), existing_aggregate.x);
	
}
float4 core_logic(in int2 vals,in float2 position, in float4 outcolors)
{
	int box_area = box_radius * box_radius;	
	float4 currentaverage = float4(0.0,0.0,0.0,0.0);
	float currentstdlum = 0.0;
	float4 temp;
	float4 temp2;
	float4 aggregatetemp = float4 (0,0,0,0);
	float xcomp;
	float ycomp;
	float zcomp;
	float2 fulloff;
	float4 result;
	int sclx = vals.x;
	int scly = vals.y;
	
	[unroll(425)] for (int i = 0; i < (box_area); i++)
	{
		fulloff.x = i / box_radius;
		fulloff.y = i % box_radius;
		fulloff.x = position.x + (fulloff.x / width)*sclx;
		fulloff.y = position.y + (fulloff.y / height)*scly;
		temp = image.Sample(linear_clamp, fulloff);
	
		aggregatetemp = welford_update(float4(aggregatetemp.xyz,RGBtoHSV(temp.xyz).z));
		xcomp = currentaverage.x + temp.x;
		ycomp = currentaverage.y + temp.y;
		zcomp = currentaverage.z + temp.z;
		currentaverage = float4 (xcomp,ycomp,zcomp,1.0);
	}
	temp = currentaverage;
	temp2 = welford_finalize(aggregatetemp);
	currentaverage = temp / temp2.w;
	// // can change to .z/.y on a whim
	currentstdlum = temp2.y;
	//result = float4(currents.xyz,currentstdlum);
	//result = float4(clamp(temp2.y,0,0.5),0.0,0.0,currentstdlum);
	// if (currentstdlum <= currents.w)
	// {
	result = float4(currentaverage.xyz,currentstdlum);
	
	// }
	return result;
	
}
float4 sortfour(in float4 val1, in float4 val2, in float4 val3, in float4 val4)
{
	float4 compare1 = val1;
	float4 compare2 = val3;
	if (compare1.w > val2.w)
	{
	compare1 = val2;
	}
	if (compare2.w > val4.w)
	{
	compare2 = val4;
	}
	if (compare1.w > compare2.w)
	{
	compare1 = compare2;
	}
	return float4(compare1.xyzw);
}
float1 weighttweak(in float1 stddev)
{
	//return float(float(1.0)/(float(1.0)+float(pow((float(255.0)*stddev*stddev),int(sharpness/int(2))))));
	return float(1.0/float(stddev));
}
float4 avgfour(in float4 val1, in float4 val2, in float4 val3, in float4 val4)
{
	float3 total;
	val1.w = weighttweak(val1.w);
	val2.w = weighttweak(val2.w);
	val3.w = weighttweak(val3.w);
	val4.w = weighttweak(val4.w);
	float totaldev = (val1.w + val2.w + val3.w + val4.w);
	total.x = clamp((val1.x*(val1.w)+val2.x*(val2.w)+val3.x*(val3.w)+val4.x*(val4.w)),0.0,4.0*totaldev);
	total.y = (val1.y*(val1.w)+val2.y*(val2.w)+val3.y*(val3.w)+val4.y*(val4.w))/float(totaldev);
	total.z = (val1.z*(val1.w)+val2.z*(val2.w)+val3.z*(val3.w)+val4.z*(val4.w))/float(totaldev);
	return (total.xyz,1.0);
}

float4 get_neighbor_regions(in float2 position)
{
	float sclx;
	float scly;
	float4 quadrant1;
	float4 quadrant2;
	float4 quadrant3;
	float4 quadrant4;
	float4 leastaverage = float4(1.0,1.0,1.0,1.0);
	float4 result;
	
	
	
	// // having to 4x this hurts but whateva
	// // // Q1
	sclx = 1.0;
	scly = 1.0;
	quadrant1.xyzw = core_logic(float2(sclx,scly),float2(position), float4(1.0,1.0,0.0,1.0));
	

	//Q2
	sclx = -1.0;
	scly = 1.0;
	quadrant2.xyzw = core_logic(float2(sclx,scly),float2(position), float4(1.0,0.0,0.0,1.0));
	
	//Q3
	sclx = -1.0;
	scly = -1.0;
	quadrant3.xyzw = core_logic(float2(sclx,scly),float2(position), float4(0.0,1.0,1.0,1.0));
	
	//Q4
	sclx = 1.0;
	scly = -1.0;
	quadrant4.xyzw = core_logic(float2(sclx,scly),float2(position), float4(0.0,0.0,1.0,1.0));
	// // end hell
	leastaverage = sortfour(quadrant1,quadrant2,quadrant3,quadrant4);
	//leastaverage = float4(quadrant1.x,quadrant2.x,quadrant3.x,quadrant4.x);
	//leastaverage = float4(
		// (quadrant1.x+quadrant2.x+quadrant3.x+quadrant4.x)/4.0,
		// (quadrant1.y+quadrant2.y+quadrant3.y+quadrant4.y)/4.0,
		// (quadrant1.z+quadrant2.z+quadrant3.z+quadrant4.z)/4.0,1.0);
	return float4(leastaverage.xyz,1.0);
	//return float4(quadrant1.w,quadrant2.w,quadrant3.w,1.0);
}


float4 pixel_shader_kuwa(pixel_data pixel) : TARGET
{
    float4 chosenkuwa = get_neighbor_regions(pixel.uv);
	//float4 chosenkuwa = float4((RGBtoHSV(float3(image.Sample(linear_clamp, pixel.uv).xyz))).zzz,1.0);
	//float4 chosenkuwa = float4(0.0,0.0,0.0,1.0)
    return float4(chosenkuwa);
}

technique Draw
{
    pass
    {
        vertex_shader = vertex_shader_kuwa(vertex);
        pixel_shader  = pixel_shader_kuwa(pixel);
    }
}
