/*
* Vulkan Demo Scene 
*
* Don't take this a an example, it's more of a personal playground
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* Note : Different license than the other examples!
*
* This code is licensed under the Mozilla Public License Version 2.0 (http://opensource.org/licenses/MPL-2.0)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "Scene.h"
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"
#include "cornellTestScene.h"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false
#define TEX_DIM 1024


class VulkanExample : public VulkanExampleBase
{
public:
	int numTilesX, numTilesY , tileWidth, tileHeight ,tileX ,tileY ;

	int current = 1;
	vks::Texture textureComputeTarget;
	
	GLSLPT::Scene *scene = nullptr;
	// Resources for the graphics part of the example
	struct Graphics {
		VkDescriptorSetLayout descriptorSetLayout;	
		VkDescriptorSet descriptorSet;				
		VkPipeline pipeline;						
		VkPipelineLayout pipelineLayout;		
	} tile,accum;

	struct {
		vks::Buffer uniformbuffer;
		
		struct StorageBuffers {
			vks::Buffer  BVHTex, BBoxminTex, BBoxmaxTex, vertexIndicesTex, verticesTex, normalsTex,
				materialsTex, transformsTex, lightsTex, textureMapsArrayTex, hdrTex, hdrMarginalDistTex, hdrConditionalDistTex;			
		} storageBuffers;
		struct Camera
		{
			glm::vec3 pos;
			float fov;
			glm::vec3 lookat;
			

		};
		struct UBO
		{  
			glm::vec3 randomVector;
			float aspectRatio;

			glm::vec2 screenResolution;
			float hdrTexSize;
			float hdrResolution;

			int numOfLights;
			int maxDepth;
			int topBVHIndex;
			int vertIndicesSize;
			int numTilesX, numTilesY, tileX, tileY;
			
			alignas(16) Camera camera;
	
		} ubo;
	}compute;
	struct UBO {
		alignas(16)int first = 0;
	}ubo;
	vks::Buffer uniformbuffer;
	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
		VkFramebuffer frameBuffer;
		
	};
	struct OffscreenPass {
		int32_t width, height;
		
		std::vector<FrameBufferAttachment> color;
		VkRenderPass renderPass;
		VkSampler colorSampler;
		
	} accumPass,tilePass;
	

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;

	
	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Path Tracing";
		
		width = 1024;
		height = 1024;
		numTilesX = 4, numTilesY =4;
		tileWidth = (int)width / numTilesX;
		tileHeight = (int)height / numTilesY;
		
		tileX = -1;
		tileY = -1;
		settings.overlay = true;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
	// Graphics
		vkDestroySampler(device, accumPass.colorSampler, nullptr);
		vkDestroySampler(device, tilePass.colorSampler, nullptr);

		// Frame buffer
		for (auto& framebuffer : accumPass.color)
		{
			// Attachments
			vkDestroyImageView(device, framebuffer.view, nullptr);
			vkDestroyImage(device, framebuffer.image, nullptr);
			vkFreeMemory(device, framebuffer.mem, nullptr);
			vkDestroyFramebuffer(device, framebuffer.frameBuffer, nullptr);
		}
		// Frame buffer
		for (auto& framebuffer : tilePass.color)
		{
			// Attachments
			vkDestroyImageView(device, framebuffer.view, nullptr);
			vkDestroyImage(device, framebuffer.image, nullptr);
			vkFreeMemory(device, framebuffer.mem, nullptr);
			vkDestroyFramebuffer(device, framebuffer.frameBuffer, nullptr);
		}
		vkDestroyRenderPass(device, accumPass.renderPass, nullptr);
		vkDestroyRenderPass(device, tilePass.renderPass, nullptr);

		vkDestroyPipeline(device, accum.pipeline, nullptr);
		vkDestroyPipeline(device, tile.pipeline, nullptr);

		vkDestroyPipelineLayout(device, accum.pipelineLayout, nullptr);
		vkDestroyPipelineLayout(device, tile.pipelineLayout, nullptr);

	    vkDestroyDescriptorSetLayout(device, accum.descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, tile.descriptorSetLayout, nullptr);


		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device,pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		compute.uniformbuffer.destroy();
		compute.storageBuffers.BBoxmaxTex.destroy();
		compute.storageBuffers.BBoxminTex.destroy();
		compute.storageBuffers.BVHTex.destroy();
		compute.storageBuffers.lightsTex.destroy();
		compute.storageBuffers.materialsTex.destroy();
		compute.storageBuffers.normalsTex.destroy();
		compute.storageBuffers.transformsTex.destroy();
		compute.storageBuffers.vertexIndicesTex.destroy();
		compute.storageBuffers.verticesTex.destroy();
		textureComputeTarget.destroy();
		

	}
	
	// Set up a separate render pass for the offscreen frame buffer
	// This is necessary as the offscreen frame buffer attachments use formats different to those from the example render pass
	void prepareOffscreenRenderpass(OffscreenPass &offscreenPass,VkFormat format,int width,int height,int framebuffercount, VkAttachmentLoadOp op)
	{
		offscreenPass.width = width;
		offscreenPass.height = height;

		// Create a separate render pass for the scene rendering as it may differ from the one used for scene rendering

		VkAttachmentDescription attchmentDescription = {};
		// Color attachment
		attchmentDescription.format = format;
		attchmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescription.loadOp = op;
		attchmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attchmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescription.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		

		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &attchmentDescription;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenPass.renderPass));

		// Create sampler to sample from the color attachments
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &offscreenPass.colorSampler));
	
		offscreenPass.color.resize(framebuffercount);
		// Create two frame buffers
		for(int i=0;i< framebuffercount;i++)
		prepareFramebuffer(offscreenPass,format,offscreenPass.color[i],width,height);
		


	}
	// Setup the offscreen framebuffer for rendering the scene from light's point-of-view to
	// The depth attachment of this framebuffer will then be used to sample from in the fragment shader of the shadowing pass
	void prepareFramebuffer(OffscreenPass &offscreenPass,VkFormat colorFormat, FrameBufferAttachment &color,int width,int height)
	{
		// Color attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = colorFormat;
		image.extent.width = width;
		image.extent.height = height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		// We will sample directly from the color attachment
		image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format = colorFormat;
		colorImageView.flags = 0;
		colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &color.image));
		

		vkGetImageMemoryRequirements(device, color.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &color.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, color.image, color.mem, 0));

		colorImageView.image = color.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &color.view));

		

		VkImageView attachments[1];
		attachments[0] = color.view;
		

		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = offscreenPass.renderPass;
		fbufCreateInfo.attachmentCount = 1;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width =width;
		fbufCreateInfo.height = height;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &color.frameBuffer));
		


	}
	// Prepare a texture target that is used to store compute shader calculations
	void prepareTextureTarget(vks::Texture *tex, uint32_t width, uint32_t height, VkFormat format)
	{
		// Get device properties for the requested texture format
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
		// Check if requested image format supports image storage operations
		assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

		// Prepare blit target texture
		tex->width = width;
		tex->height = height;

		VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = { width, height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// Image will be sampled in the fragment shader and used as storage target in the compute shader
		imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		imageCreateInfo.flags = 0;

		VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &tex->image));
		vkGetImageMemoryRequirements(device, tex->image, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &tex->deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, tex->image, tex->deviceMemory, 0));

		VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		tex->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		vks::tools::setImageLayout(
			layoutCmd,
			tex->image,
			VK_IMAGE_ASPECT_COLOR_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			tex->imageLayout);

		vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);

		// Create sampler
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.compareOp = VK_COMPARE_OP_NEVER;
		sampler.minLod = 0.0f;
		sampler.maxLod = 0.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &tex->sampler));

		// Create image view
		VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view.format = format;
		view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
		view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		view.image = tex->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &tex->view));

		// Initialize a descriptor for later use
		tex->descriptor.imageLayout = tex->imageLayout;
		tex->descriptor.imageView = tex->view;
		tex->descriptor.sampler = tex->sampler;
		tex->device = vulkanDevice;
		//setTexture(tex);
	}
	// Setup and fill the compute shader storage buffers containing primitives for the raytraced scene
	void prepareStorageBuffers(unsigned int NUMBER,unsigned int UNIT_SIZE,void* data,vks::Buffer & storage_buffer)
	{
		
		VkDeviceSize storageBufferSize =NUMBER * UNIT_SIZE;
	
		// Stage
		vks::Buffer stagingBuffer;

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			storageBufferSize,
			data);

		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&storage_buffer,
			storageBufferSize);

		// Copy to staging buffer
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storage_buffer.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		stagingBuffer.destroy();
		
	}

	

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			{
				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				VkClearValue clearValues[1];
				clearValues[0].color = defaultClearColor;
				renderPassBeginInfo.renderPass = tilePass.renderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = tileWidth;
				renderPassBeginInfo.renderArea.extent.height = tileWidth;
				renderPassBeginInfo.clearValueCount = 1;
				renderPassBeginInfo.pClearValues = clearValues;
				// Set target frame buffer
				renderPassBeginInfo.framebuffer = tilePass.color[0].frameBuffer;

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
				VkViewport viewport = vks::initializers::viewport((float)tileWidth, (float)tileWidth, 0.0f, 1.0f);

				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				VkRect2D scissor = vks::initializers::rect2D(tileWidth, tileWidth, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
				
				// Display ray traced image generated by compute shader as a full screen quad
				// Quad vertices are generated in the vertex shader
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, tile.pipelineLayout, 0, 1, &tile.descriptorSet, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, tile.pipeline);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}
			{
				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				VkClearValue clearValues[1];
				clearValues[0].color = defaultClearColor;
				renderPassBeginInfo.renderPass = accumPass.renderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = 1;
				renderPassBeginInfo.pClearValues = clearValues;
				renderPassBeginInfo.framebuffer = accumPass.color[1 - current].frameBuffer;
				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
				// Set target frame buffer
				if (tileX == 0 && tileY == 0) {
					VkRect2D range = vks::initializers::rect2D(width, height, 0, 0);
					VkClearRect clearrect = { range,0,1 };
					VkClearAttachment clearattachment = { VK_IMAGE_ASPECT_COLOR_BIT ,1,defaultClearColor };
					vkCmdClearAttachments(drawCmdBuffers[i], 1, &clearattachment, 1, &clearrect);
				}
			
				vkCmdEndRenderPass(drawCmdBuffers[i]);

			}
			{
				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				VkClearValue clearValues[1];
				clearValues[0].color = defaultClearColor;
				renderPassBeginInfo.renderPass = accumPass.renderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = 1;
				renderPassBeginInfo.pClearValues = clearValues;
				
				renderPassBeginInfo.framebuffer = accumPass.color[current].frameBuffer;

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
				//VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);

				VkViewport viewport = vks::initializers::viewport((float)tileWidth, (float)tileHeight, 0.0f, 1.0f);
				viewport.x = tileX * tileWidth;
				viewport.y= tileY * tileHeight;
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
			
				// Display ray traced image generated by compute shader as a full screen quad
				// Quad vertices are generated in the vertex shader
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, accum.pipelineLayout, 0, 1, &accum.descriptorSet, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, accum.pipeline);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}

			{
				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				VkClearValue clearValues[2];
				clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[1].depthStencil = { 1.0f, 0 };
				renderPassBeginInfo.renderPass = renderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = 2;
				renderPassBeginInfo.pClearValues = clearValues;
				// Set target frame buffer
				renderPassBeginInfo.framebuffer = frameBuffers[i];

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
				

				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				// Display ray traced image generated by compute shader as a full screen quad
				// Quad vertices are generated in the vertex shader
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				drawUI(drawCmdBuffers[i]);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}
			

			

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}

	}

	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),			// Compute UBO
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5),	// Graphics image samplers
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),				// Storage image for ray traced image output
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10),			// Storage buffer for the scene primitives
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				3);

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	
	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
				0,
				VK_FALSE);

		VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				VK_POLYGON_MODE_FILL,
				VK_CULL_MODE_FRONT_BIT,
				VK_FRONT_FACE_COUNTER_CLOCKWISE,
				0);

		VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

		VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_FALSE,
				VK_FALSE,
				VK_COMPARE_OP_LESS_OR_EQUAL);

		VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				VK_SAMPLE_COUNT_1_BIT,
				0);

		std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				dynamicStateEnables.size(),
				0);

		// Display pipeline
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/vulkanscene/texture.vert", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/vulkanscene/pt.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				tile.pipelineLayout,
				tilePass.renderPass,
				0);

		VkPipelineVertexInputStateCreateInfo emptyInputState{};
		emptyInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		emptyInputState.vertexAttributeDescriptionCount = 0;
		emptyInputState.pVertexAttributeDescriptions = nullptr;
		emptyInputState.vertexBindingDescriptionCount = 0;
		emptyInputState.pVertexBindingDescriptions = nullptr;
		pipelineCreateInfo.pVertexInputState = &emptyInputState;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = tilePass.renderPass;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &tile.pipeline));

		shaderStages[0] = loadShader(getAssetPath() + "shaders/vulkanscene/texture.vert", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/vulkanscene/appear.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

		pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				accum.pipelineLayout,
				accumPass.renderPass,
				0);
		pipelineCreateInfo.pVertexInputState = &emptyInputState;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = accumPass.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &accum.pipeline));

		shaderStages[0] = loadShader(getAssetPath() + "shaders/vulkanscene/texture.vert", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/vulkanscene/texture.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

		pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass,
				0);
		pipelineCreateInfo.pVertexInputState = &emptyInputState;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));


	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&compute.uniformbuffer,
			sizeof(compute.ubo));
		VK_CHECK_RESULT(compute.uniformbuffer.map());

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformbuffer,
			sizeof(ubo));
		VK_CHECK_RESULT(uniformbuffer.map());

		updateUniformBuffers();
	}
	
	void updateUniformBuffers()
	{
		
		tileX++;
		
		if (tileX == numTilesX)
		{
			tileY++;
			if (tileY == numTilesY)
			{
				current = 1 - current;
				tileY = 0;
				ubo.first++;
				updateDescriptorSet();
				updateTileDescriptorSet();

			}
			tileX = 0;
		}
		
		
	    compute.ubo.aspectRatio = (float)width / (float)height;
		compute.ubo.camera.fov = scene->camera->fov;
		compute.ubo.camera.lookat = scene->camera->pivot;
		compute.ubo.camera.pos = scene->camera->position;
		compute.ubo.maxDepth =1;
		compute.ubo.numTilesX = numTilesX;
		compute.ubo.numTilesY = numTilesY;
		compute.ubo.tileX = tileX;
		compute.ubo.tileY = tileY;

		compute.ubo.numOfLights = scene->lights.size();
		float r1 = ((float)rand() / (RAND_MAX));
		float r2 = ((float)rand() / (RAND_MAX));
		float r3 = ((float)rand() / (RAND_MAX));
		compute.ubo.randomVector = glm::vec3(r1, r2, r3);
		compute.ubo.screenResolution = glm::vec2(width, height);
		compute.ubo.topBVHIndex =  scene->bvhTranslator.topLevelIndexPackedXY;
		memcpy(uniformbuffer.mapped, &ubo, sizeof(ubo));

		memcpy(compute.uniformbuffer.mapped, &compute.ubo, sizeof(compute.ubo));
	}

	void loadScene()
	{
		delete scene;
		//scene = LoadScene(std::string("./assets/")+sceneFilenames[index]);
		scene = new GLSLPT::Scene();
		
		loadCornellTestScene(scene);
		
		
		if (!scene)
		{
			std::cout << "Unable to load scene\n";
			exit(0);
		}
		std::cout << "Scene Loaded\n\n";
	}
	void prepareSSBO() {
		
		prepareStorageBuffers(scene->bvhTranslator.nodes.size(), sizeof(glm::ivec4), scene->bvhTranslator.nodes.data(), compute.storageBuffers.BVHTex);
		prepareStorageBuffers(scene->bvhTranslator.bboxmin.size(), sizeof(glm::vec4), scene->bvhTranslator.bboxmin.data(), compute.storageBuffers.BBoxminTex);
		prepareStorageBuffers(scene->bvhTranslator.bboxmax.size(), sizeof(glm::vec4), scene->bvhTranslator.bboxmax.data(), compute.storageBuffers.BBoxmaxTex);
		prepareStorageBuffers(scene->vertIndices.size(), sizeof(glm::ivec4), scene->vertIndices.data(), compute.storageBuffers.vertexIndicesTex);
		prepareStorageBuffers(scene->vertices_uvx.size(), sizeof(glm::vec4), scene->vertices_uvx.data(), compute.storageBuffers.verticesTex);
		prepareStorageBuffers(scene->normals_uvy.size(), sizeof(glm::vec4), scene->normals_uvy.data(), compute.storageBuffers.normalsTex);
		prepareStorageBuffers(scene->materials.size(), sizeof(GLSLPT::Material), scene->materials.data(), compute.storageBuffers.materialsTex);
		prepareStorageBuffers(scene->transforms.size(), sizeof(glm::mat4), scene->transforms.data(), compute.storageBuffers.transformsTex);
		prepareStorageBuffers(scene->lights.size(), sizeof(GLSLPT::Light), scene->lights.data(), compute.storageBuffers.lightsTex);

	}
	// Prepare the compute pipeline that generates the ray traced image
	void prepareAccum()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Storage image (raytraced output)
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				0)			
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &accum.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&accum.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &accum.pipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&accum.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &accum.descriptorSet));

		updateAccumDescriptorSet();

	}
	void updateAccumDescriptorSet() {

		VkDescriptorImageInfo ColorDescriptor =
			vks::initializers::descriptorImageInfo(
				tilePass.colorSampler,
				tilePass.color[0].view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0: Output storage image
			vks::initializers::writeDescriptorSet(
				accum.descriptorSet,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				0,
				&ColorDescriptor),
		

		};

		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
	}
	// Prepare the compute pipeline that generates the ray traced image
	void prepareTile()
	{
	

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Storage image (raytraced output)
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				0),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				1),
			// Binding 1: Uniform buffer block
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				2),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				3),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				4),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				5),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				6),

			
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				7),
			
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				8),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				9),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				10),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				11),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				12),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				13),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				14),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				15),
			
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &tile.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&tile.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &tile.pipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&tile.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &tile.descriptorSet));

		updateTileDescriptorSet();

	}
	
	void updateTileDescriptorSet(){

		VkDescriptorImageInfo ColorDescriptor =
			vks::initializers::descriptorImageInfo(
				accumPass.colorSampler,
				accumPass.color[1 - current].view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0: Output storage image
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				0,
				&ColorDescriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				1,
				&textureComputeTarget.descriptor),
			// Binding 1: Uniform buffer block
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				2,
				&compute.uniformbuffer.descriptor),
			// Binding 2: Shader storage buffer for the spheres
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				7,
				&compute.storageBuffers.BVHTex.descriptor),
			// Binding 2: Shader storage buffer for the planes
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				8,
				&compute.storageBuffers.BBoxminTex.descriptor),

			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				9,
				&compute.storageBuffers.BBoxmaxTex.descriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				10,
				&compute.storageBuffers.vertexIndicesTex.descriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				11,
				&compute.storageBuffers.verticesTex.descriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				12,
				&compute.storageBuffers.normalsTex.descriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				13,
				&compute.storageBuffers.materialsTex.descriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				14,
				&compute.storageBuffers.transformsTex.descriptor),
			vks::initializers::writeDescriptorSet(
				tile.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				15,
				&compute.storageBuffers.lightsTex.descriptor),
		
		};

		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
	}
	void setupDescriptorSetLayout()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
		{ vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				0),
			// Binding 0 : Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				1)
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));
	}

	void setupDescriptorSet()
	{
		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		updateDescriptorSet();

	}
	void updateDescriptorSet() {

		VkDescriptorImageInfo ColorDescriptor =
			vks::initializers::descriptorImageInfo(
				accumPass.colorSampler,
				accumPass.color[1-current].view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		std::vector<VkWriteDescriptorSet> writeDescriptorSets =
		{ 
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				0,
				&uniformbuffer.descriptor),
			
			// Binding 0 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				1,
				&ColorDescriptor)
		};

		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
	}
	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
		
	}

	
	void prepare()
	{
		
		VulkanExampleBase::prepare();
		loadScene();
		prepareSSBO();
	
		prepareUniformBuffers();
		
		prepareOffscreenRenderpass(tilePass,VK_FORMAT_R32G32B32A32_SFLOAT,tileWidth,tileHeight,1, VK_ATTACHMENT_LOAD_OP_CLEAR);
		prepareTextureTarget(&textureComputeTarget, TEX_DIM, TEX_DIM, VK_FORMAT_R32G32B32A32_SFLOAT);
		prepareOffscreenRenderpass(accumPass, VK_FORMAT_R32G32B32A32_SFLOAT, width, height, 2, VK_ATTACHMENT_LOAD_OP_DONT_CARE);

		setupDescriptorSetLayout();
		setupDescriptorPool();
		setupDescriptorSet();

		prepareTile();
		prepareAccum();

		preparePipelines();
		
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		
		draw();
		
		updateUniformBuffers();
	
	   buildCommandBuffers();
		
		
	  
		
	}
	

	virtual void viewChanged()
	{
		updateUniformBuffers();
		
		buildCommandBuffers();
		
	}

};

std::shared_ptr<VulkanExample> vulkanExample;                                                                   \
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)						\
{																									\
if (vulkanExample != NULL)																		\
{																								\
vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);									\
}																								\
return (DefWindowProc(hWnd, uMsg, wParam, lParam));												\
}																									\
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)									\
{																									\
for (int32_t i = 0; i < __argc; i++) { VulkanExample::args.push_back(__argv[i]); };  			\
	vulkanExample.reset(new VulkanExample());															\
	vulkanExample->initVulkan();																	\
	vulkanExample->setupWindow(hInstance, WndProc);													\
	vulkanExample->prepare();																		\
	vulkanExample->renderLoop();																	\

	return 0;																						\
}