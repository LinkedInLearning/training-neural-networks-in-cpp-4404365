/*
  ==============================================================================

   This file is part of the JUCE tutorials.
   Copyright (c) 2020 - Raw Material Software Limited

   The code included in this file is provided under the terms of the ISC license
   http://www.isc.org/downloads/software-support-policy/isc-license. Permission
   To use, copy, modify, and/or distribute this software for any purpose with or
   without fee is hereby granted provided that the above copyright notice and
   this permission notice appear in all copies.

   THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES,
   WHETHER EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR
   PURPOSE, ARE DISCLAIMED.

  ==============================================================================
*/

/*******************************************************************************
 The block below describes the properties of this PIP. A PIP is a short snippet
 of code that can be read by the Projucer and used to generate a JUCE project.

 BEGIN_JUCE_PIP_METADATA

 name:             SDRNN_7to1
 version:          1.0.0
 vendor:           JUCE
 website:          http://juce.com
 description:      Segment Display Recognition Neural Network (7 to 1 model).

 dependencies:     juce_core, juce_data_structures, juce_events, juce_graphics,
                   juce_gui_basics
 exporters:        xcode_mac, vs2019, linux_make

 type:             Component
 mainClass:        MainContentComponent

 useLocalCopy:     1

 END_JUCE_PIP_METADATA

*******************************************************************************/


#pragma once
#include "MLP.h"
#include <string>

#define WIN_W 500
#define WIN_H 350

#define O1X  20
#define O1Y  20
#define VSW 72
#define VSL 100
#define HSW 50
#define HSL 100
#define V1X 0+O1X
#define V2X V1X+VSW+VSL
#define H1Y 0+O1Y
#define H2Y H1Y+HSW+VSL-15
#define H3Y H2Y+HSW+VSL-15
#define SLaX V1X+VSW 
#define SLaY H1Y
#define SLbX V2X
#define SLbY H1Y+HSW
#define SLcX V2X
#define SLcY H2Y+HSW
#define SLdX V1X+VSW 
#define SLdY H3Y
#define SLeX V1X
#define SLeY H2Y+HSW
#define SLfX V1X
#define SLfY H1Y+HSW
#define SLgX V1X+VSW 
#define SLgY H2Y
#define SEGW 25
#define SEGL 100
#define aX SLaX
#define aY SLaY+((HSW*5)/12)
#define bX SLbX
#define bY SLbY
#define cX SLcX
#define cY SLcY
#define dX SLdX
#define dY SLdY+((HSW*5)/12)
#define eX SLeX+((VSW*5)/8)
#define eY SLeY
#define fX SLfX+((VSW*5)/8)
#define fY SLfY
#define gX SLgX
#define gY SLgY+((HSW*5)/12)

#define OFFSET 230

//==============================================================================
class MainContentComponent : public juce::Component,
                             public juce::Slider::Listener,
                             public juce::Button::Listener,
                             public juce::TextEditor::Listener{
public:
    //==============================================================================
    MainContentComponent(){
        srand(time(NULL));
        rand();
        sdrnn = new MultiLayerPerceptron({7,7,1});
        LookAndFeel *l = &getLookAndFeel();
        l->setColour(juce::Slider::thumbColourId,juce::Colour(110,110,110));
        l->setColour(juce::Slider::textBoxOutlineColourId,Colour(240,240,240));
        l->setColour(juce::Slider::textBoxTextColourId,juce::Colours::black);
        l->setColour(juce::Slider::backgroundColourId,juce::Colour(OFFSET,OFFSET,OFFSET));
        l->setColour(juce::Slider::trackColourId, juce::Colour(OFFSET, OFFSET, OFFSET));
        l->setColour(juce::TextButton::buttonColourId, juce::Colour(OFFSET, OFFSET, OFFSET));
        l->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
        l->setColour(juce::TextButton::textColourOnId, juce::Colours::black);
        l->setColour(juce::Label::textColourId, juce::Colours::black);
        l->setColour(juce::Label::backgroundColourId, juce::Colour(240,240,240));
        l->setColour(juce::TextEditor::ColourIds::textColourId, juce::Colours::black);
        l->setColour(juce::TextEditor::ColourIds::backgroundColourId, juce::Colours::white);

        addAndMakeVisible(lbl_epochs_txt);
        addAndMakeVisible(entry_epochs);
        lbl_epochs_txt.setText("Epochs to Train:", no);
        entry_epochs.setText("10");
        
        addAndMakeVisible(btn_train);
        btn_train.setButtonText("Train some more");
        btn_train.addListener(this);

        addAndMakeVisible(lbl_err_txt);
        addAndMakeVisible(lbl_err);
        lbl_err_txt.setText("Training Error:",no);
        lbl_err.setText("---",no);
        lbl_err.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_tepochs_txt);
        addAndMakeVisible(lbl_tepochs);
        lbl_tepochs_txt.setText("Epochs so far:",no);
        lbl_tepochs.setText("0",no);
        lbl_tepochs.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_out);
        addAndMakeVisible(lbl_out_txt);
        lbl_out_txt.setText("Raw Output:",no);
        lbl_out.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_int);
        addAndMakeVisible(lbl_int_txt);
        lbl_int_txt.setText("Number Output:", no);
        lbl_int.setFont(60);
        lbl_int.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(btn_reset);
        btn_reset.setButtonText("Reset");
        btn_reset.addListener(this);

        addAndMakeVisible (slider_a);
        slider_a.setRange (0.0, 1.0);
        slider_a.addListener (this);
        slider_a.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_a.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_b);
        slider_b.setRange(0.0,1.0);
        slider_b.addListener(this);
        slider_b.setSliderStyle(juce::Slider::LinearVertical);
        slider_b.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_b.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_c);
        slider_c.setRange(0.0,1.0);
        slider_c.addListener(this);
        slider_c.setSliderStyle(juce::Slider::LinearVertical);
        slider_c.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_c.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_d);
        slider_d.setRange(0.0,1.0);
        slider_d.addListener(this);
        slider_d.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_d.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_e);
        slider_e.setRange(0.0,1.0);
        slider_e.addListener(this);
        slider_e.setSliderStyle(juce::Slider::LinearVertical);
        slider_e.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_e.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_f);
        slider_f.setRange(0.0,1.0);
        slider_f.addListener(this);
        slider_f.setSliderStyle(juce::Slider::LinearVertical);
        slider_f.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_f.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_g);
        slider_g.setRange(0.0,1.0);
        slider_g.addListener(this);
        slider_g.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_g.setNumDecimalPlacesToDisplay(2);

        setSize(WIN_W, WIN_H);
        run_ann();
    }


#define DX   90
#define DY   30
#define LEN  90
#define HEI  25 
#define X0  290
#define X1 (X0+DX)
#define Y0   30
#define Y1 (Y0+DY)
#define Y2 (Y1+DY)
#define Y3 (Y2+DY)
#define Y4 (Y3+DY)
#define Y5 (Y4+DY)
#define YN (Y5-10)
#define YR (Y5+DY)

    void resized() override{
        slider_a.setBounds(SLaX,SLaY,HSL,HSW);
        slider_b.setBounds(SLbX,SLbY,VSW,VSL);
        slider_c.setBounds(SLcX,SLcY,VSW,VSL);
        slider_d.setBounds(SLdX,SLdY,HSL,HSW);
        slider_e.setBounds(SLeX,SLeY,VSW,VSL);
        slider_f.setBounds(SLfX,SLfY,VSW,VSL);
        slider_g.setBounds(SLgX,SLgY,HSL,HSW);

        lbl_epochs_txt.setBounds  (X0, Y0, LEN, HEI);
        entry_epochs.setBounds    (X1, Y0, LEN, HEI);
        btn_train.setBounds       (X1, Y1, LEN, HEI);
        lbl_err_txt.setBounds     (X0, Y2, LEN, HEI);
        lbl_err.setBounds         (X1, Y2, LEN, HEI);
        lbl_tepochs_txt.setBounds (X0, Y3, LEN, HEI);
        lbl_tepochs.setBounds     (X1, Y3, LEN, HEI);
        lbl_out_txt.setBounds     (X0, Y4, LEN, HEI);
        lbl_out.setBounds         (X1, Y4, LEN, HEI);
        lbl_int_txt.setBounds     (X0, Y5, LEN, HEI);
        lbl_int.setBounds         (X1, YN, LEN, 50);
        btn_reset.setBounds       (X0, YR, LEN, HEI);
       
        setSize(WIN_W,WIN_H);
    }

    void sliderValueChanged (juce::Slider* slider) override {
        Colour c(    (int)(slider->getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider->getValue()* OFFSET),
            OFFSET - (int)(slider->getValue()* OFFSET)    );

        slider->setColour(juce::Slider::backgroundColourId,c);
        slider->setColour(juce::Slider::trackColourId,c);
        run_ann();
        repaint();
    }

    void run_ann(){
        std::vector <double> x;
        x.push_back(slider_a.getValue());
        x.push_back(slider_b.getValue());
        x.push_back(slider_c.getValue());
        x.push_back(slider_d.getValue());
        x.push_back(slider_e.getValue());
        x.push_back(slider_f.getValue());
        x.push_back(slider_g.getValue());
        double theoutput = sdrnn->run(x)[0];
        lbl_out.setText(std::to_string(theoutput),no);
        lbl_int.setText(std::to_string(std::min((int)(theoutput * 10), 9)),no);
    }

    void train_ann(){
        double MSE;
        int epochs = entry_epochs.getText().getIntValue();
        for (int i = 0; i < epochs; i++){
            MSE = 0.0;
            MSE += sdrnn->bp({1,1,1,1,1,1,0}, {0.05}); //0 pattern
            MSE += sdrnn->bp({0,1,1,0,0,0,0}, {0.15}); //1 pattern
            MSE += sdrnn->bp({1,1,0,1,1,0,1}, {0.25}); //2 pattern
            MSE += sdrnn->bp({1,1,1,1,0,0,1}, {0.35}); //3 pattern
            MSE += sdrnn->bp({0,1,1,0,0,1,1}, {0.45}); //4 pattern
            MSE += sdrnn->bp({1,0,1,1,0,1,1}, {0.55}); //5 pattern
            MSE += sdrnn->bp({1,0,1,1,1,1,1}, {0.65}); //6 pattern
            MSE += sdrnn->bp({1,1,1,0,0,0,0}, {0.75}); //7 pattern
            MSE += sdrnn->bp({1,1,1,1,1,1,1}, {0.85}); //8 pattern
            MSE += sdrnn->bp({1,1,1,1,0,1,1}, {0.95}); //9 pattern
        }
        MSE /= 10.0;
        lbl_err.setText(std::to_string(MSE), no);
        tepochs += epochs;
        lbl_tepochs.setText(std::to_string(tepochs), no);
        run_ann();
    }

    void buttonClicked(juce::Button* button) override{
        if (button == &btn_train)
            train_ann();
        if (button == &btn_reset){
            delete(sdrnn);
            sdrnn = new MultiLayerPerceptron({7,7,1});
            tepochs = 0;
            lbl_err.setText("---", no);
            lbl_tepochs.setText(std::to_string(tepochs), no);
            run_ann();
        }
    }

    void paint(juce::Graphics& g) override {
        g.fillAll(juce::Colour(240,240,240));
        
        g.setColour(juce::Colour(
            (int)(slider_a.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_a.getValue()* OFFSET),
            OFFSET - (int)(slider_a.getValue()* OFFSET)     ));
        g.fillRect(aX,aY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_b.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_b.getValue()* OFFSET),
            OFFSET - (int)(slider_b.getValue()* OFFSET)     ));
        g.fillRect(bX,bY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_c.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_c.getValue()* OFFSET),
            OFFSET - (int)(slider_c.getValue()* OFFSET)     ));
        g.fillRect(cX,cY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_d.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_d.getValue()* OFFSET),
            OFFSET - (int)(slider_d.getValue()* OFFSET)     ));
        g.fillRect(dX,dY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_e.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_e.getValue()* OFFSET),
            OFFSET - (int)(slider_e.getValue()* OFFSET)     ));
        g.fillRect(eX,eY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_f.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_f.getValue()* OFFSET),
            OFFSET - (int)(slider_f.getValue()* OFFSET)     ));
        g.fillRect(fX,fY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_g.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_g.getValue()* OFFSET),
            OFFSET - (int)(slider_g.getValue()* OFFSET)     ));
        g.fillRect(gX,gY,SEGL,SEGW);

    }

    MultiLayerPerceptron * sdrnn;
    juce::NotificationType no = juce::dontSendNotification;
    int tepochs = 0;

private:
    juce::Slider slider_a;
    juce::Slider slider_b;
    juce::Slider slider_c;
    juce::Slider slider_d;
    juce::Slider slider_e;
    juce::Slider slider_f;
    juce::Slider slider_g;

    juce::Label lbl_epochs_txt;
    juce::TextEditor entry_epochs;
    juce::TextButton btn_train;
    juce::Label lbl_err_txt; 
    juce::Label lbl_err;
    juce::Label lbl_tepochs_txt;
    juce::Label lbl_tepochs;
    juce::Label lbl_out_txt;
    juce::Label lbl_out;
    juce::Label lbl_int_txt;
    juce::Label lbl_int;
    juce::TextButton btn_reset;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainContentComponent)
};
